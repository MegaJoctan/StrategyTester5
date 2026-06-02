from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
# from threading import Lock
import time
import glob
import os
import logging
from typing import Iterable, Optional, Literal, Any
from pathlib import Path
from choreographer.channels import JSONError
from ..MetaTrader5 import AccountInfo, SymbolInfo, TerminalInfo, TICKS_DTYPE, RATES_DTYPE, MULTI_TICKS_DTYPE, \
    MULTI_RATES_DTYPE
from strategytester5.MetaTrader5.constants import MetaTrader5Constants
# import MetaTrader5 as mt5
import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from strategytester5 import config


def _warning_log(msg: str, logger: Optional[logging.Logger] = None):
    if logger is None:
        print(msg)
        return

    logger.warning(msg)


class HistoryManager:
    def __init__(self, mt5_instance: Optional[Any] = None,
                 broker_data_path: Optional[str] = config.DEFAULT_BROKER_DATA_PATH,
                 logger: Optional[logging.Logger] = None):

        """
        A class for managing historical market data for the simulation. It provides methods for synchronizing bars and ticks from the MetaTrader5 terminal, as well as loading them from locally stored parquet files.
        Args:
            mt5_instance (MetaTrader5) : Initialized MetaTrader5 instance to extract bars from
            broker_data_path (Optional |str): A directory where the synchronized bars and ticks will be stored as parquet files. Defaults to `strategytester5.config.DEFAULT_BROKER_DATA_PATH`
            logger (Optional[logging.Logger], optional): The logger to use. Defaults to None.
        """

        self.mt5_instance = mt5_instance
        self.broker_data_dir = broker_data_path
        self.logger = logger
        self._last_start_date: int
        self._last_end_date: int

        # self._mt5_lock = Lock()

    @staticmethod
    def bars_file_path(symbol: str, timeframe_str: str, year: int, month: int,
                       broker_data_dir: Optional[str] = config.DEFAULT_BROKER_DATA_PATH) -> Path:
        return Path(broker_data_dir) / symbol / "Bars" / timeframe_str / f"{year:04d}{month:02d}.parquet"

    @staticmethod
    def ticks_file_path(symbol: str, year: int, month: int,
                        broker_data_dir: Optional[str] = config.DEFAULT_BROKER_DATA_PATH) -> Path:
        return Path(broker_data_dir) / symbol / "Ticks" / f"{year:04d}{month:02d}.parquet"

    @staticmethod
    def month_start(month: int, year: int) -> datetime:
        return datetime(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def next_month(month: int, year: int) -> datetime:
        dt = HistoryManager.month_start(month, year)
        if dt.month == 12:
            return dt.replace(year=dt.year + 1, month=1)
        return dt.replace(month=dt.month + 1)

    @staticmethod
    def month_floor(dt: datetime) -> datetime:
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def next_month_dt(dt: datetime) -> datetime:
        dt = HistoryManager.month_floor(dt)
        if dt.month == 12:
            return dt.replace(year=dt.year + 1, month=1)
        return dt.replace(month=dt.month + 1)

    @staticmethod
    def iter_months_between(date_from: datetime, date_to: datetime) -> Iterable[tuple[int, int]]:
        cur = HistoryManager.month_floor(date_from)

        while cur <= date_to:
            yield cur.year, cur.month
            cur = HistoryManager.next_month_dt(cur)

    def _info_log(self, msg: str):
        if self.logger is None:
            print(msg)
            return

        self.logger.info(msg)

    def _error_log(self, msg: str):
        if self.logger is None:
            print(msg)
            return

        self.logger.error(msg)

    def _warning_log(self, msg: str):
        if self.logger is None:
            print(msg)
            return

        self.logger.warning(msg)

    def _critical_log(self, msg: str):
        if self.logger is None:
            print(msg)
            return

        self.logger.critical(msg)

    def synchronize_bars(self,
                         symbol: str,
                         timeframe: int,
                         month: int,
                         year: int,
                         ) -> Optional[pl.DataFrame]:
        """
        Extracts bars (rates) from the MetaTrader 5 terminal, stores them in a nearby location for simulator usage.

        Returns:
            Synchronized bars in a polars DataFrame object

        Args:
            timeframe (int) : A timeframe to extract bars from
            symbol (str) : An instrument in the terminal
            month (int) : The entire month to synchronize
            year (int) : A year which a specified `month` belongs
        """

        tf_str = MetaTrader5Constants.TIMEFRAME2STRING_MAP[timeframe]

        if self.mt5_instance is None:
            log = ("Cannot synchronize Bars from MetaTrader5, due to an invalid MetaTrader5 instance.\n"
                   "If the default MetaTrader5 wasn't installed initially, run `pip install strategytester5[mt5]`")

            self._critical_log(log)
            raise RuntimeError(log)

        start = self.month_start(month, year)
        end = self.next_month(month, year)

        # add it to the market watch
        if not self.mt5_instance.symbol_select(symbol, True):
            err = f"Failed to select or add {symbol} to the MarketWatch, mt5 error = {self.mt5_instance.last_error()}"
            self._error_log(err)
            return None

        self._info_log(f"Fetching bars from MetaTrader5 on: {tf_str} from {symbol} for: {year:04d}-{month:02d}")

        # with self._mt5_lock:
        rates = self.mt5_instance.copy_rates_range(symbol, timeframe, start, end)
        if rates is None or len(rates) == 0:
            warn = f"No bars were received from MetaTrader5 on: {tf_str} from {symbol} for: {year:04d}-{month:02d} : start = {start}, end = {end}"
            self._warning_log(warn)
            return None

        # rates dataframe

        pd_df = pd.DataFrame(rates)
        df = pl.DataFrame(pd_df)
        # df = pl.DataFrame(rates)

        file = self.bars_file_path(symbol=symbol, timeframe_str=tf_str,
                                   year=year, month=month,
                                   broker_data_dir=self.broker_data_dir)

        file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(file)

        self._info_log(f"---> {tf_str} Synchronized!")

        return df

    def synchronize_ticks(
            self,
            symbol: str,
            month: int,
            year: int,
    ) -> Optional[pl.DataFrame]:
        """
        Extracts ticks from the MetaTrader 5 terminal, stores them in a nearby location for simulator usage.

        Returns:
            Synchronized ticks in a polars DataFrame object

        Args:
            symbol (str) : An instrument in the terminal
            month (int) : The entire month to synchronize
            year (int) : A year which a specified `month` belongs
        """

        if self.mt5_instance is None:
            log = ("Cannot synchronize ticks from MetaTrader5, due to an invalid MetaTrader5 instance.\n"
                   "If the default MetaTrader5 wasn't installed initially, run `pip install strategytester5[mt5]`")

            self._critical_log(log)
            raise RuntimeError(log)

        start = self.month_start(month, year)
        end = self.next_month(month, year)

        # add it to the market watch
        if not self.mt5_instance.symbol_select(symbol, True):
            err = f"Failed to select or add {symbol} to the MarketWatch, mt5 error = {self.mt5_instance.last_error()}"
            self._error_log(err)
            return None

        self._info_log(f"Fetching ticks from MetaTrader5 from {symbol} for: {year:04d}-{month:02d}")

        # with self._mt5_lock:
        ticks = self.mt5_instance.copy_ticks_range(symbol, start, end, self.mt5_instance.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            warn = f"No ticks were received from MetaTrader5 from {symbol} for: {year:04d}-{month:02d}"
            self._warning_log(warn)
            return None

        # rates dataframe
        df = pl.DataFrame(ticks)

        file = self.ticks_file_path(symbol, year=year, month=month, broker_data_dir=self.broker_data_dir)
        file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(file)

        return df

    def synchronize_all_timeframes(
            self,
            symbols: Iterable[str],
            date_from: datetime,
            date_to: datetime,
            max_workers: int = 4,
    ):

        """
        Synchronizes bars from all timeframes from the MetaTrader5 terminal for specified symbols and date range. This method is useful for pre-populating the local storage with bars from all timeframes, so the simulator can later load them without accessing the terminal.

        Args:
            symbols (Iterable[str]): A list of symbols to synchronize
            date_from (datetime): The start date for synchronization
            date_to (datetime): The end date for synchronization
            max_workers (int): The maximum number of worker threads to use

        """

        if date_from > date_to:
            self._warning_log("date_from must be <= date_to")
            return

        all_tfs = list(MetaTrader5Constants.STRING2TIMEFRAME_MAP.values())

        self._info_log("Synchronizing all timeframes...")

        tasks = []

        # Build all jobs (symbol x timeframe x year x month)
        for symbol in symbols:
            for tf in all_tfs:
                for year, month in self.iter_months_between(date_from, date_to):
                    tasks.append((symbol, tf, year, month))

        start = time.time()

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            futures = [
                executor.submit(self.synchronize_bars, sym, tf, month, year)
                for sym, tf, year, month in tasks
            ]

            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)

                # sym, tf, y, m, status = res

        elapsed = time.time() - start
        self._info_log(f"Finished synchronization in {elapsed:.2f}s")

    @staticmethod
    def copy_rates_range_from_parquet(
            symbol: str,
            timeframe: int,
            date_from: datetime,
            date_to: datetime,
            polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto",
            broker_data_dir: Optional[str] = config.DEFAULT_BROKER_DATA_PATH,
            logger: Optional[logging.Logger] = None
    ):
        """Copies bars (rates) for a specified symbol, timeframe and date range from locally stored parquet files. This method is used by the simulator to load bars without accessing the MetaTrader5 terminal.

        Args:
            symbol (str): An instrument in the terminal
            timeframe (int): A timeframe to extract bars from
            date_from (datetime): start date of the bars to copy
            date_to (datetime): end date of the bars to copy
            polars_collect_engine (Literal["auto", "in-memory", "streaming", "gpu"], optional): Polars collection engine to use. Defaults to "auto".
            broker_data_dir (Optional[str], optional): The directory where broker data is stored. Defaults to config.DEFAULT_BROKER_DATA_PATH.
            logger (Optional[logging.Logger], optional): The logger to use. Defaults to None.

        Returns:
            Copied bars as a NumPy array or None in case of a failure
        """

        if date_from > date_to:
            _warning_log("date_from must be <= date_to", logger)
            return None

        if isinstance(date_from, (int, float)):
            date_from = datetime.fromtimestamp(date_from)
        if isinstance(date_to, (int, float)):
            date_to = datetime.fromtimestamp(date_to)

        timeframe_str = MetaTrader5Constants.TIMEFRAME2STRING_MAP[timeframe]

        files: list[str] = []
        for year, month in HistoryManager.iter_months_between(date_from, date_to):
            file = HistoryManager.bars_file_path(symbol=symbol, timeframe_str=timeframe_str, year=year, month=month,
                                                 broker_data_dir=broker_data_dir)
            if file.exists():
                files.append(str(file))

        if not files:
            _warning_log(
                f"No stored bar history found for {symbol} {timeframe_str} "
                f"between {date_from} and {date_to}", logger
            )
            return None

        t_from = int(date_from.timestamp())
        t_to = int(date_to.timestamp())

        try:
            df = (
                pl.scan_parquet(files)
                .filter(
                    (pl.col("time") >= t_from) &
                    (pl.col("time") <= t_to)
                )
                .sort("time")
                .select([
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "tick_volume",
                    "spread",
                    "real_volume",
                ])
                .collect(engine=polars_collect_engine)
            )

            if df.is_empty():
                return np.empty(0, dtype=RATES_DTYPE)

            rows = list(
                zip(
                    df["time"].to_list(),
                    df["open"].to_list(),
                    df["high"].to_list(),
                    df["low"].to_list(),
                    df["close"].to_list(),
                    df["tick_volume"].to_list(),
                    df["spread"].to_list(),
                    df["real_volume"].to_list(),
                )
            )

            return np.array(rows, dtype=RATES_DTYPE)

        except Exception as e:
            _warning_log(
                f"Failed to copy stored rates for {symbol} {timeframe_str} "
                f"from {date_from} to {date_to}: {e}", logger
            )
            return None

    @staticmethod
    def copy_rates_from_parquet(
            symbol: str,
            timeframe: int,
            date_from: datetime,
            history_start_date: datetime,
            count: int,
            polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto",
            broker_data_dir: Optional[str] = config.DEFAULT_BROKER_DATA_PATH,
            logger: Optional[logging.Logger] = None
    ):
        """Copies bars (rates) for a specified symbol, timeframe and date range from locally stored parquet files. This method is used by the simulator to load bars without accessing the MetaTrader5 terminal.

        Args:
            symbol (str): An instrument in the terminal
            timeframe (int): A timeframe to extract bars from
            date_from (datetime): start date of the bars to copy
            history_start_date (datetime): The first date in history
            count (int | optional): The maximum number of ticks to collect
            polars_collect_engine (Literal["auto", "in-memory", "streaming", "gpu"], optional): Polars collection engine to use. Defaults to "auto".
            broker_data_dir (Optional[str], optional): The directory where broker data is stored. Defaults to config.DEFAULT_BROKER_DATA_PATH.
            logger (Optional[logging.Logger], optional): The logger to use. Defaults to None.

        Returns:
            Copied bars as a NumPy array or None in case of a failure
        """

        timeframe_str = MetaTrader5Constants.TIMEFRAME2STRING_MAP[timeframe]

        if isinstance(date_from, (int, float)):
            date_from = datetime.fromtimestamp(date_from)
        if isinstance(history_start_date, (int, float)):
            history_start_date = datetime.fromtimestamp(history_start_date)

        # ---------------- determine months ----------------

        start_dt = min(date_from, history_start_date)
        end_dt = max(date_from, history_start_date)

        months_iter = (HistoryManager.iter_months_between(start_dt, end_dt))

        files: list[str] = []
        for year, month in months_iter:
            file = HistoryManager.bars_file_path(
                symbol=symbol,
                timeframe_str=timeframe_str,
                year=year,
                month=month,
                broker_data_dir=broker_data_dir
            )

            if file.exists():
                files.append(str(file))
            else:
                _warning_log(f"{file} not found, skipping", logger)

        if not files:
            _warning_log(
                f"No stored bar history found for {symbol} {timeframe_str} searched paths: {files}",
                logger
            )
            return None

        t_from = int(date_from.timestamp())
        try:
            lf = pl.scan_parquet(files)

            # ---------------- sorting ----------------

            lf = lf.filter(pl.col("time") <= t_from)
            lf = lf.sort("time", descending=True)
            lf = lf.limit(count)

            df = (
                lf.select([
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "tick_volume",
                    "spread",
                    "real_volume",
                ])
                .collect(engine=polars_collect_engine)
            )

            if df.is_empty():
                return np.empty(0, dtype=RATES_DTYPE)

            # ---------------- convert ----------------
            arr = np.array(
                list(zip(
                    df["time"],
                    df["open"],
                    df["high"],
                    df["low"],
                    df["close"],
                    df["tick_volume"],
                    df["spread"],
                    df["real_volume"],
                )),
                dtype=RATES_DTYPE
            )

            return arr[::-1]

        except Exception as e:
            _warning_log(
                f"Failed to copy stored rates for {symbol} {timeframe_str} "
                f"from {date_from}: {e}",
                logger
            )
            return None

    @staticmethod
    def _tick_flag_mask(flags: int) -> int:
        if flags == MetaTrader5Constants.COPY_TICKS_ALL:
            return (
                    MetaTrader5Constants.TICK_FLAG_BID
                    | MetaTrader5Constants.TICK_FLAG_ASK
                    | MetaTrader5Constants.TICK_FLAG_LAST
                    | MetaTrader5Constants.TICK_FLAG_VOLUME
                    | MetaTrader5Constants.TICK_FLAG_BUY
                    | MetaTrader5Constants.TICK_FLAG_SELL
            )

        mask = 0
        if flags & MetaTrader5Constants.COPY_TICKS_INFO:
            mask |= MetaTrader5Constants.TICK_FLAG_BID | MetaTrader5Constants.TICK_FLAG_ASK
        if flags & MetaTrader5Constants.COPY_TICKS_TRADE:
            mask |= MetaTrader5Constants.TICK_FLAG_LAST | MetaTrader5Constants.TICK_FLAG_VOLUME

        return mask

    @staticmethod
    def copy_ticks_range_parquet(
            symbol: str,
            date_from: datetime,
            date_to: Optional[datetime] = None,
            flags: int = MetaTrader5Constants.COPY_TICKS_ALL,
            limit: Optional[int] = None,
            polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto",
            broker_data_dir: Optional[str] = config.DEFAULT_BROKER_DATA_PATH,
            logger: Optional[logging.Logger] = None,
    ):

        """Copies ticks for a specified symbol, timeframe and date range from locally stored parquet files. This method is used by the simulator to load ticks without accessing the MetaTrader5 terminal.

        Args:
            symbol (str): An instrument in the terminal
            date_from (datetime): start date of the ticks to copy
            date_to (datetime): end date of the ticks to copy
            flags (int): A flag to define the type of the requested ticks. COPY_TICKS_INFO – ticks with Bid and/or Ask changes, COPY_TICKS_TRADE – ticks with changes in Last and Volume, COPY_TICKS_ALL – all ticks. For any type of request, the values of the previous tick are added to the remaining fields of the MqlTick structure.
            limit (int | optional): The maximum number of ticks to collect
            polars_collect_engine (Literal["auto", "in-memory", "streaming", "gpu"], optional): Polars collection engine to use. Defaults to "auto".
            broker_data_dir (Optional[str], optional): The directory where broker data is stored. Defaults to config.DEFAULT_BROKER_DATA_PATH.
            logger (Optional[logging.Logger], optional): The logger to use. Defaults to None.

        Returns:
            Copied ticks as a NumPy array or None in case of a failure
        """

        if limit is None:
            if date_to is None:
                _warning_log("Either date_to or limit must be provided", logger)
                return None

            if date_from > date_to:
                _warning_log("date_from must be <= date_to", logger)
                return None

        if isinstance(date_from, (int, float)):
            date_from = datetime.fromtimestamp(date_from)
        if isinstance(date_to, (int, float)):
            date_to = datetime.fromtimestamp(date_to)

        # ---------------- determine months ----------------
        months_iter = (
            HistoryManager.iter_months_between(date_from, date_to)
            if limit is None
            else HistoryManager.iter_months_between(date_from, date_from + timedelta(days=365 * 20))
            # wide range fallback
        )

        files: list[str] = []
        for year, month in months_iter:
            file = HistoryManager.ticks_file_path(symbol, year, month, broker_data_dir)
            if file.exists():
                files.append(str(file))

        if not files:
            _warning_log(f"No stored tick history found for {symbol}", logger)
            return None

        t_from = int(date_from.timestamp())
        t_to = int(date_to.timestamp()) if date_to else None

        flag_mask = HistoryManager._tick_flag_mask(flags)

        try:
            lf = pl.scan_parquet(files)

            # ---------------- filtering ----------------
            lf = lf.filter(pl.col("time") >= t_from)

            if limit is None:
                lf = lf.filter(pl.col("time") <= t_to)

            lf = lf.filter((pl.col("flags") & flag_mask) != 0)

            # ---------------- sorting ----------------
            lf = lf.sort(["time", "time_msc"])

            # ---------------- limit mode ----------------
            if limit is not None:
                lf = lf.limit(limit)

            df = (
                lf.select([
                    "time",
                    "bid",
                    "ask",
                    "last",
                    "volume",
                    "time_msc",
                    "flags",
                    "volume_real",
                ])
                .collect(engine=polars_collect_engine)
            )

            if df.is_empty():
                return np.empty(0, dtype=TICKS_DTYPE)

            return np.array(
                list(zip(
                    df["time"],
                    df["bid"],
                    df["ask"],
                    df["last"],
                    df["volume"],
                    df["time_msc"],
                    df["flags"],
                    df["volume_real"],
                )),
                dtype=TICKS_DTYPE
            )

        except Exception as e:
            _warning_log(
                f"Failed to copy ticks for {symbol} from {date_from}: {e}",
                logger
            )
            return None

    @staticmethod
    def copy_ticks_from_parquet(
            symbol: str,
            date_from: datetime,
            flags: int = MetaTrader5Constants.COPY_TICKS_ALL,
            limit: Optional[int] = None,
            polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto",
            broker_data_dir: Optional[str] = config.DEFAULT_BROKER_DATA_PATH,
            logger: Optional[logging.Logger] = None,
    ):

        """Copies ticks for a specified symbol, timeframe and date range from locally stored parquet files. This method is used by the simulator to load ticks without accessing the MetaTrader5 terminal.

        Args:
            symbol (str): An instrument in the terminal
            date_from (datetime): start date of the ticks to copy
            flags (int): A flag to define the type of the requested ticks. COPY_TICKS_INFO – ticks with Bid and/or Ask changes, COPY_TICKS_TRADE – ticks with changes in Last and Volume, COPY_TICKS_ALL – all ticks. For any type of request, the values of the previous tick are added to the remaining fields of the MqlTick structure.
            limit (int | optional): The maximum number of ticks to collect
            polars_collect_engine (Literal["auto", "in-memory", "streaming", "gpu"], optional): Polars collection engine to use. Defaults to "auto".
            broker_data_dir (Optional[str], optional): The directory where broker data is stored. Defaults to config.DEFAULT_BROKER_DATA_PATH.
            logger (Optional[logging.Logger], optional): The logger to use. Defaults to None.

        Returns:
            Copied ticks as a NumPy array or None in case of a failure
        """

        # ---------------- locate parquet files ----------------

        files = sorted(
            glob.glob(
                os.path.join(
                    broker_data_dir,
                    symbol,
                    "Ticks",
                    "*.parquet"
                ),
                recursive=True
            )
        )

        if not files:
            _warning_log(f"No stored tick history found for {symbol}", logger)
            return None

        if isinstance(date_from, (int, float)):
            date_from = datetime.fromtimestamp(date_from)

        t_from = int(date_from.timestamp())
        flag_mask = HistoryManager._tick_flag_mask(flags)

        try:
            lf = pl.scan_parquet(files)

            # ---------------- filtering ----------------

            lf = lf.filter(pl.col("time") >= t_from)
            lf = lf.filter((pl.col("flags") & flag_mask) != 0)

            # ---------------- sorting ----------------

            lf = lf.sort(["time", "time_msc"])

            # ---------------- limit mode ----------------
            if limit is not None:
                lf = lf.limit(limit)

            df = (
                lf.select([
                    "time",
                    "bid",
                    "ask",
                    "last",
                    "volume",
                    "time_msc",
                    "flags",
                    "volume_real",
                ])
                .collect(engine=polars_collect_engine)
            )

            if df.is_empty():
                return np.empty(0, dtype=TICKS_DTYPE)

            return np.array(
                list(zip(
                    df["time"],
                    df["bid"],
                    df["ask"],
                    df["last"],
                    df["volume"],
                    df["time_msc"],
                    df["flags"],
                    df["volume_real"],
                )),
                dtype=TICKS_DTYPE
            )

        except Exception as e:
            _warning_log(
                f"Failed to copy ticks for {symbol} from {date_from}: {e}",
                logger
            )
            return None

    def _ensure_ticks_month(self, symbol, year, month, sync: bool):
        file = self.ticks_file_path(symbol, year, month, self.broker_data_dir)

        if file.exists():
            return True

        if not sync:
            return False

        df = self.synchronize_ticks(symbol, month, year)
        return df is not None

    def _process_symbol_ticks(
            self,
            symbol: str,
            symbol_id: int,
            date_from: datetime,
            date_to: datetime,
            sync: bool,
            polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto"
    ) -> Optional[np.ndarray]:

        files = []

        for year, month in self.iter_months_between(date_from, date_to):

            if not self._ensure_ticks_month(symbol, year, month, sync):
                continue

            file = self.ticks_file_path(symbol, year, month, self.broker_data_dir)
            files.append(str(file))

        if not files:
            return None

        t_from = int(date_from.timestamp())
        t_to = int(date_to.timestamp())

        df = (
            pl.scan_parquet(files)
            .filter(
                (pl.col("time") >= t_from) &
                (pl.col("time") <= t_to)
            )
            .collect(engine=polars_collect_engine)
        )

        if df.is_empty():
            return None

        df = df.with_columns(
            pl.lit(symbol_id).alias("symbol_id")
        )

        return np.array(
            list(zip(
                df["time"].to_list(),
                df["bid"].to_list(),
                df["ask"].to_list(),
                df["last"].to_list(),
                df["volume"].to_list(),
                df["time_msc"].to_list(),
                df["flags"].to_list(),
                df["volume_real"].to_list(),
                df["symbol_id"].to_list(),
            )),
            dtype=MULTI_TICKS_DTYPE
        )

    def _ensure_bars_month(self, symbol: str, timeframe: int, year: int, month: int, sync: bool):
        file = self.bars_file_path(
            symbol,
            MetaTrader5Constants.TIMEFRAME2STRING_MAP[timeframe],
            year,
            month,
            self.broker_data_dir,
        )

        if file.exists():
            return True

        if not sync:
            return False

        df = self.synchronize_bars(symbol, timeframe, month, year)
        return df is not None

    def _process_symbol_bars(
            self,
            symbol: str,
            symbol_id: int,
            timeframe: int,
            date_from: datetime,
            date_to: datetime,
            sync: bool,
            polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto"
    ) -> Optional[np.ndarray]:

        tf_str = MetaTrader5Constants.TIMEFRAME2STRING_MAP[timeframe]

        files = []

        for year, month in self.iter_months_between(date_from, date_to):

            if not self._ensure_bars_month(symbol, timeframe, year, month, sync):
                continue

            file = self.bars_file_path(symbol, tf_str, year, month, self.broker_data_dir)
            files.append(str(file))

        if not files:
            return None

        t_from = int(date_from.timestamp())
        t_to = int(date_to.timestamp())

        df = (
            pl.scan_parquet(files)
            .filter(
                (pl.col("time") >= t_from) &
                (pl.col("time") <= t_to)
            )
            .collect(engine=polars_collect_engine)
        )

        if df.is_empty():
            return None

        df = df.with_columns(
            pl.lit(symbol_id).alias("symbol_id")
        )

        return np.array(
            list(zip(
                df["time"].to_list(),
                df["open"].to_list(),
                df["high"].to_list(),
                df["low"].to_list(),
                df["close"].to_list(),
                df["tick_volume"].to_list(),
                df["spread"].to_list(),
                df["real_volume"].to_list(),
                df["symbol_id"].to_list(),
            )),
            dtype=MULTI_RATES_DTYPE
        )

    def load_ticks_lazy_multi(
            self,
            symbols: list[str],
            date_from: datetime,
            date_to: datetime,
            sync: bool = True,
    ) -> pl.LazyFrame:

        symbol_to_id = {s: i for i, s in enumerate(symbols)}

        lazy_frames = []

        for symbol in symbols:
            files = []

            for year, month in self.iter_months_between(date_from, date_to):
                if not self._ensure_ticks_month(symbol, year, month, sync):
                    continue

                file = self.ticks_file_path(symbol, year, month, self.broker_data_dir)
                files.append(str(file))

            if not files:
                continue

            lf = (
                pl.scan_parquet(files)
                .filter(
                    (pl.col("time") >= int(date_from.timestamp())) &
                    (pl.col("time") <= int(date_to.timestamp()))
                )
                .with_columns(pl.lit(symbol_to_id[symbol]).alias("symbol_id"))
            )

            lazy_frames.append(lf)

        if not lazy_frames:
            err = "No tick data found"
            self._critical_log(err)
            raise RuntimeError(err)

        return pl.concat(lazy_frames)

    def build_tick_stream(
            self,
            symbols: list[str],
            date_from: datetime,
            date_to: datetime,
            sync: bool = True,
            polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto"
    ) -> pl.DataFrame:

        """Builds a tick stream for specified symbols and date range. This method is used by the simulator to load ticks without accessing the MetaTrader5 terminal.

        Args:
            symbols (list[str]): A list of symbols to load ticks for
            date_from (datetime): start date of the ticks to load
            date_to (datetime): end date of the ticks to load
            sync (bool): whether to synchronize missing months from the terminal. Defaults to True.
            polars_collect_engine (Literal["auto", "in-memory", "streaming", "gpu"], optional): Polars collection engine to use. Defaults to "auto".
        """

        lf = self.load_ticks_lazy_multi(symbols, date_from, date_to, sync)

        df = (
            lf
            .sort(["time", "time_msc"])  # global ordering
            .select([
                "time",
                "bid",
                "ask",
                "last",
                "volume",
                "time_msc",
                "flags",
                "volume_real",
                "symbol_id",
            ])
            .collect(engine=polars_collect_engine)  # ⚡ critical
        )

        return df

    def load_bars_lazy_multi(
            self,
            symbols: list[str],
            timeframe: int,
            date_from: datetime,
            date_to: datetime,
            sync: bool = True,
    ) -> pl.LazyFrame:

        symbol_to_id = {s: i for i, s in enumerate(symbols)}
        tf_str = MetaTrader5Constants.TIMEFRAME2STRING_MAP[timeframe]

        lazy_frames = []

        for symbol in symbols:
            files = []

            for year, month in self.iter_months_between(date_from, date_to):
                if not self._ensure_bars_month(symbol, timeframe, year, month, sync):
                    continue

                file = self.bars_file_path(symbol, tf_str, year, month, self.broker_data_dir)
                files.append(str(file))

            if not files:
                continue

            lf = (
                pl.scan_parquet(files)
                .filter(
                    (pl.col("time") >= int(date_from.timestamp())) &
                    (pl.col("time") <= int(date_to.timestamp()))
                )
                .with_columns(pl.lit(symbol_to_id[symbol]).alias("symbol_id"))
            )

            lazy_frames.append(lf)

        if not lazy_frames:
            err = f"No bars data found on: {tf_str} timeframe"
            self._critical_log(err)
            raise RuntimeError(err)

        return pl.concat(lazy_frames)

    def build_bar_stream(
            self,
            symbols: list[str],
            timeframe: int,
            date_from: datetime,
            date_to: datetime,
            sync: bool = True,
            polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto"
    ) -> pl.DataFrame:

        """Builds a bar stream for specified symbols and date range. This method is used by the simulator to load bars without accessing the MetaTrader5 terminal.

        Args:
            symbols (list[str]): A list of symbols to load bars for
            timeframe (int): The timeframe for the bars
            date_from (datetime): start date of the bars to load
            date_to (datetime): end date of the bars to load
            sync (bool): whether to synchronize missing months from the terminal. Defaults to True.
            polars_collect_engine (Literal["auto", "in-memory", "streaming", "gpu"], optional): Polars collection engine to use. Defaults to "auto".
        """

        lf = self.load_bars_lazy_multi(symbols, timeframe, date_from, date_to, sync)

        df = (
            lf
            .sort(["time", "symbol_id"])  # bars don't need time_msc
            .select([
                "time",
                "open",
                "high",
                "low",
                "close",
                "tick_volume",
                "spread",
                "real_volume",
                "symbol_id",
            ])
            .collect(engine=polars_collect_engine)
        )

        return df


def import_account_info(path: Path, logger: Optional[logging.Logger] = None) -> Optional[AccountInfo]:
    if not path.is_file() or not path.exists():
        err = "Broker path does not exist"
        if logger:
            logger.error(err)

        raise FileNotFoundError(err)

    try:
        with path.open(encoding="utf-8") as json_file:
            data = json.load(json_file)

        account_info = AccountInfo(**data["account_info"])

    except Exception as e:
        err = f"Failed to read a file {path} JSON error = {e}"
        if logger:
            logger.error(err)

        raise JSONError(err)

    return account_info


def import_all_symbol_info(path: Path, logger: Optional[logging.Logger] = None) -> Optional[SymbolInfo]:
    if not path.is_file() or not path.exists():
        err = "Broker path does not exist"
        if logger:
            logger.error(err)

        raise FileNotFoundError(err)

    try:
        with path.open(encoding="utf-8") as json_file:
            data = json.load(json_file)

        all_symbol_info = []

        for s in data["all_symbols_info"]:
            all_symbol_info.append(SymbolInfo(**s))

    except Exception as e:
        err = f"Failed to read a file {path} JSON error = {e}"
        if logger:
            logger.error(err)

        raise JSONError(err)

    return tuple(all_symbol_info)


def _ensure_dir(path: Path, logger: Optional[logging.Logger] = None):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        if logger:
            logger.error(f"Failed to create broker path: {e}")
        raise


def _to_dict(obj):
    """Safely convert namedtuple/dataclass/dict to dict"""
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return dict(obj)


def export_account_info(
        ac_info: AccountInfo | dict,
        file: Path,
        logger: Optional[logging.Logger] = None,
) -> bool:
    file.parent.mkdir(parents=True, exist_ok=True)

    try:
        payload = {
            "account_info": _to_dict(ac_info),
        }

        with file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        if logger:
            logger.error(f"Export account_info failed: {e}")
        return False


def export_all_symbol_info(
        all_symbol_info: Iterable[SymbolInfo | dict],
        file: Path,
        logger: Optional[logging.Logger] = None,
) -> bool:
    file.parent.mkdir(parents=True, exist_ok=True)

    try:
        payload = {
            "all_symbols_info": [_to_dict(s) for s in all_symbol_info],
        }

        with file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        if logger:
            logger.error(f"Export symbol_info failed: {e}")
        return False


def export_terminal_info(
        terminal_info: TerminalInfo | dict,
        file: Path,
        logger: Optional[logging.Logger] = None,
) -> bool:
    file.parent.mkdir(parents=True, exist_ok=True)

    try:
        payload = {
            "terminal_info": _to_dict(terminal_info),
        }

        with file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        if logger:
            logger.error(f"Export terminal_info failed: {e}")
        return False


def import_terminal_info(
        path: Path,
        logger: Optional[logging.Logger] = None,
) -> TerminalInfo:
    if not path.is_file():
        err = "Terminal info file does not exist"
        if logger:
            logger.error(err)
        raise FileNotFoundError(err)

    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        return TerminalInfo(**data["terminal_info"])

    except Exception as e:
        err = f"Failed to read {path} JSON error = {e}"
        if logger:
            logger.error(err)
        raise JSONError(err)

