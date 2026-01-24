from strategytester5 import STRING2TIMEFRAME_MAP, TIMEFRAME2STRING_MAP
from strategytester5.hist import ticks, bars
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
import time
import os
from typing import Any, Self
from typing import Optional
import logging

class HistoryManager:
    def __init__(self,
                mt5_instance: Any,
                symbols: list,
                start_dt: datetime,
                end_dt: datetime,
                timeframe: int,
                LOGGER: Optional[logging.Logger] = None,
                max_fetch_workers: int=None,
                max_cpu_workers: int=None,
                history_dir: str = "History"
                ):

        """Initialize a history manager for fetching and storing MT5 data.

        Args:
            mt5_instance: MT5 API/client instance used to fetch data.
            symbols: List of symbol strings to retrieve history for.
            start_dt: Inclusive start datetime for the history window.
            end_dt: Inclusive end datetime for the history window.
            timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_M1).
            max_fetch_workers: Max concurrent fetch workers; defaults based on symbol count.
            max_cpu_workers: Max CPU workers for processing; defaults to (CPU count - 1).
            history_dir: Directory name for persisted history data.
        """

        self.mt5_instance = mt5_instance
        self.symbols = symbols
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.max_fetch_workers = max_fetch_workers
        self.max_cpu_workers = max_cpu_workers
        self.LOGGER = LOGGER
        self.history_dir = history_dir
        self.timeframe = timeframe
        
        if max_fetch_workers is None:
            self.max_fetch_workers = min(32, max(4, len(self.symbols)))
        if max_cpu_workers is None:
            self.max_cpu_workers = max(1, os.cpu_count() - 1)
    
    def __critical_log(self, msg: str):
        """Log a critical message via LOGGER or print as fallback."""
        if self.LOGGER is not None:
            self.LOGGER.critical(msg)
        else:
            print(msg)
    
    def __info_log(self, msg: str):
        """Log an info message via LOGGER or print as fallback."""
        if self.LOGGER is not None:
            self.LOGGER.info(msg)
        else:
            print(msg)
            
    def _fetch_bars_worker(self, symbol: str, timeframe: int, return_df: bool=False) -> dict:
        """Fetch historical bars for a symbol and return summary info when requested.

        Args:
            symbol: Instrument symbol to fetch.
            timeframe: MT5 timeframe constant to query.
            return_df: If True, return a dict with bars and metadata; else {}.
        """

        bars_obtained = bars.fetch_historical_bars(
            which_mt5=self.mt5_instance,
            symbol=symbol,
            timeframe=timeframe,
            start_datetime=self.start_dt,
            end_datetime=self.end_dt,
            return_df=return_df,
            LOGGER=self.LOGGER,
            hist_dir=self.history_dir
        )
        
        bars_info = {
            "symbol": symbol,
            "bars": bars_obtained,
            "size": bars_obtained.height,
            "counter": 0
        }
        
        return bars_info if return_df else {}

    def _fetch_ticks_worker(self, symbol: str, return_df: bool=False) -> dict:
        """Fetch real ticks for a symbol and return summary info when requested.

        Args:
            symbol: Instrument symbol to fetch.
            return_df: If True, return a dict with ticks and metadata; else {}.
        """
        
        ticks_obtained = ticks.fetch_historical_ticks(
            which_mt5=self.mt5_instance,
            start_datetime=self.start_dt,
            end_datetime=self.end_dt,
            symbol=symbol,
            return_df=True,
            LOGGER=self.LOGGER,
            hist_dir=self.history_dir
        )
        
        ticks_info = {
            "symbol": symbol,
            "ticks": ticks_obtained,
            "size": ticks_obtained.height,
            "counter": 0
        }

        return ticks_info if return_df else {}

    def __init_worker_mt5(self):
        """Initialize worker MT5 instance. Useful for ProcessPool"""
        import MetaTrader5 as mt5

        if not mt5.initialize():
            raise RuntimeError(f"MT5 init failed in worker: {mt5.last_error()}")

        self.mt5_instance = mt5

    def _gen_ticks_worker(self, symbol: str, symbol_points: float, return_df: bool=False) -> dict:
        """Generate synthetic ticks from M1 bars for a symbol and saves data.

        Args:
            symbol: Instrument symbol to generate ticks for.
            return_df: If True, return a dict with ticks and metadata; else {}.
        """

        one_minute_bars = bars.fetch_historical_bars(
            which_mt5=self.mt5_instance,
            symbol=symbol,
            timeframe=STRING2TIMEFRAME_MAP["M1"],  # <- use your map key directly
            start_datetime=self.start_dt,
            end_datetime=self.end_dt,
            LOGGER=self.LOGGER,
            hist_dir=self.history_dir,
            return_df=return_df
        )

        if one_minute_bars is None:
            return {}

        ticks_df = ticks.TicksGen.generate_ticks_from_bars(
            bars=one_minute_bars,
            symbol=symbol,
            symbol_point=symbol_points,
            LOGGER=self.LOGGER,
            hist_dir=self.history_dir,
            return_df=True
        )

        ticks_info = {
            "symbol": symbol,
            "ticks": ticks_df,
            "size": ticks_df.height,
            "counter": 0
        }

        return ticks_info if return_df else {}

    def fetch_history(self, modelling: str, symbol_info_func: any):
        """Fetch bars or ticks for all symbols according to the modelling mode.

        Args:
            modelling: One of "real_ticks", "every_tick", "new_bar", "1-minute-ohlc".

        Returns:
            Tuple of (all_bars_info, all_ticks_info) lists.
        """

        all_ticks_info = []
        all_bars_info = []
        
        if modelling == "real_ticks":
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=self.max_fetch_workers) as executor:
                futs = {executor.submit(self._fetch_ticks_worker, s, True): s for s in self.symbols}

                for fut in as_completed(futs):
                    sym = futs[fut]
                    try:
                        res = fut.result()              # <- get dict
                        all_ticks_info.append(res)
                    except Exception as e:
                        self.__critical_log(f"Failed to fetch real ticks for {sym}: {e!r}")

            total_ticks = sum(info["size"] for info in all_ticks_info)
            self.__info_log(f"Total real ticks collected: {total_ticks} in {(time.time()-start_time):.2f} seconds.")

        elif modelling == "every_tick":

            start_time = time.time()

            with ThreadPoolExecutor(max_workers=self.max_fetch_workers) as executor:
                futs = {executor.submit(self._gen_ticks_worker,s, symbol_info_func(s).point, True): s for s in self.symbols}

                for fut in as_completed(futs):
                    sym = futs[fut]
                    try:
                        res = fut.result()              # <- get dict
                        all_ticks_info.append(res)
                    except Exception as e:
                        self.__critical_log(f"Failed to generate ticks for {sym}: {e!r}")

            total_ticks = sum(info["size"] for info in all_ticks_info)
            self.__info_log(f"Total ticks generated: {total_ticks} in {(time.time()-start_time):.2f} seconds.")

        elif modelling in ("new_bar", "1-minute-ohlc"):
            
            start_time = time.time()
            tf = STRING2TIMEFRAME_MAP["M1"] if modelling == "1-minute-ohlc" else STRING2TIMEFRAME_MAP[self.timeframe]
            
            with ThreadPoolExecutor(max_workers=self.max_fetch_workers) as executor:
                futs = {executor.submit(self._fetch_bars_worker, s, tf, True): s for s in self.symbols}

                for fut in as_completed(futs):
                    sym = futs[fut]
                    try:
                        res = fut.result()              # <- get dict
                        all_bars_info.append(res)
                    except Exception as e:
                        self.__critical_log(f"Failed to fetch bars for {sym}: {e!r}")

            total_bars = sum(info["size"] for info in all_bars_info)
            self.__info_log(f"Total bars collected: {total_bars} from '{TIMEFRAME2STRING_MAP[tf]}' timeframe in {(time.time()-start_time):.2f} seconds.")
            
        return all_bars_info, all_ticks_info

    def synchronize_timeframes(self):

        all_tfs = list(STRING2TIMEFRAME_MAP.values())
        start = time.time()

        with ThreadPoolExecutor(max_workers=self.max_fetch_workers) as ex:
            futs = {ex.submit(self._fetch_bars_worker, sym, tf, False): (sym, tf)
                    for sym in self.symbols
                    for tf in all_tfs}

            for fut in as_completed(futs):
                sym, tf = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    self.__critical_log(f"sync failed {sym} {TIMEFRAME2STRING_MAP.get(tf, tf)}: {e!r}")

        self.__info_log(f"Timeframes synchronization complete! {(time.time() - start):.2f}s elapsed.")
