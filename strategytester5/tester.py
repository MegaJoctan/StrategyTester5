from fontTools.misc.bezierTools import epsilon

from concurrent.futures import ThreadPoolExecutor, as_completed
from strategytester5 import *
from . import error_description
from datetime import datetime, timedelta
import secrets
import os
import numpy as np
import fnmatch
from typing import Optional
import polars as pl
from strategytester5.validators.trade import TradeValidators
from strategytester5.validators.tester_configs import TesterConfigValidators
from strategytester5._template import html_report_template
from strategytester5.hist.manager import  HistoryManager
import sys

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["agg.path.chunksize"] = 10000
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 0.1


class StrategyTester:
    def __init__(self,
                 tester_config: dict,
                 mt5_instance: MetaTrader5,
                 logs_dir: Optional[str]="Logs",
                 reports_dir: Optional[str]="Reports",
                 history_dir: Optional[str]="History"):
        
        """MetaTrader 5-Like Strategy tester for the MetaTrader5-Python module.

        Args:
            tester_config: Dictionary of tester configuration values.
            mt5_instance: MetaTrader5 API/client instance used for obtaining crucial information from the broker as an attempt to mimic the terminal.
            logs_dir: Directory for log files.
            reports_dir: Directory for HTML reports and assets.
            history_dir: Directory for historical data storage.
        Raises:
            RuntimeError: If required MT5 account info cannot be obtained.
        """
        
        self.reports_dir = reports_dir
        self.history_dir = history_dir
        
        self.symbol_info_cache: dict[str, SymbolInfo] = {}
        self.trade_validators_cache: dict[str, TradeValidators] = {}
        self.tick_cache: dict[str, Tick] = {}
        
        # ---------------- validate all configs from a dictionary -----------------
        
        self.tester_config = TesterConfigValidators.parse_tester_configs(tester_config)
        
        # -------------- Check if we are not on the tester mode ------------------
        
        self.IS_TESTER = not any(arg.startswith("--mt5") for arg in sys.argv) # are we on the strategy tester mode or live trading
            
        # -------------------- initialize the Loggers ----------------------------
        
        self.mt5_instance = mt5_instance
        self.simulator_name = self.tester_config["bot_name"]
        
        
        os.makedirs(logs_dir, exist_ok=True)
        self.logger = get_logger(self.simulator_name+ ".tester" if self.IS_TESTER else ".mt5", 
                                        logfile=os.path.join(logs_dir, f"{LOG_DATE}.log"),
                                        level=logging_level)
        
        global LOGGER
        LOGGER = self.logger
        
        if not self.IS_TESTER:
            self.logger.debug("MT5 mode")
            
        # -------------- Check if we are not on the tester mode ------------------
        
        self.IS_TESTER = not any(arg.startswith("--mt5") for arg in sys.argv) # are we on the strategy tester mode or live trading
        
        if not self.IS_TESTER:
            self.logger.debug("MT5 mode")
            
        # --------------- initialize ticks or bars data ----------------------------
        
        self.logger.info("StrategyTester Initializing")
        self.logger.info(f"StrategyTester configs: {self.tester_config}")

        hist_manager = HistoryManager(mt5_instance=self.mt5_instance,
                                      symbols=self.tester_config["symbols"],
                                      start_dt=self.tester_config["start_date"],
                                      end_dt=self.tester_config["end_date"],
                                      timeframe=self.tester_config["timeframe"],
                                      history_dir=self.history_dir,
                                      LOGGER=self.logger
                                      )

        for symbol in self.tester_config["symbols"]:
            self.symbol_info(symbol)

        self.TESTER_ALL_BARS_INFO, self.TESTER_ALL_TICKS_INFO = hist_manager.fetch_history(
            self.tester_config["modelling"],
            symbol_info_func=self.symbol_info,
        )

        if self.IS_TESTER:
            hist_manager.synchronize_timeframes()

        self.logger.info("Initialized")
        
        # ----------------- initialize internal containers -----------------
        
        self.AccountInfo = AccountInfo
        
        self.__orders_container__ = []
        self.__orders_history_container__ = []
        self.__positions_container__ = []
        self.__deals_history_container__ = []

        # ----------------- AccountInfo -----------------
        
        mt5_acc_info = mt5_instance.account_info()

        if mt5_acc_info is None:
            raise RuntimeError("Failed to obtain MT5 account info")

        deposit = self.tester_config["deposit"]
        
        self.__account_state_update(
            account_info=AccountInfo(
                # ---- identity / broker-controlled ----
                login=11223344,
                trade_mode=mt5_acc_info.trade_mode,
                leverage=int(self.tester_config["leverage"]),
                limit_orders=mt5_acc_info.limit_orders,
                margin_so_mode=mt5_acc_info.margin_so_mode,
                trade_allowed=mt5_acc_info.trade_allowed,
                trade_expert=mt5_acc_info.trade_expert,
                margin_mode=mt5_acc_info.margin_mode,
                currency_digits=mt5_acc_info.currency_digits,
                fifo_close=mt5_acc_info.fifo_close,

                # ---- simulator-controlled financials ----
                balance=deposit,                # simulator starting balance
                credit=mt5_acc_info.credit,
                profit=0.0,
                equity=deposit,
                margin=0.0,
                margin_free=deposit,
                margin_level=0.0,

                # ---- risk thresholds (copied from broker) ----
                margin_so_call=mt5_acc_info.margin_so_call,
                margin_so_so=mt5_acc_info.margin_so_so,
                margin_initial=mt5_acc_info.margin_initial,
                margin_maintenance=mt5_acc_info.margin_maintenance,

                # ---- rarely used but keep parity ----
                assets=mt5_acc_info.assets,
                liabilities=mt5_acc_info.liabilities,
                commission_blocked=mt5_acc_info.commission_blocked,

                # ---- descriptive ----
                name="John Doe",
                server="MetaTrader5-Simulator",
                currency=mt5_acc_info.currency,
                company=mt5_acc_info.company,
            )
        )

        self.positions_unrealized_pl = 0
        self.positions_total_margin = 0

        # -------------------- tester reports ----------------------------
        
        self.last_curve_minute = -1
        
        self.tester_curves = {
            "time": [],
            "balance": [],
            "equity": [],
            "margin": []
        }
        
        self.tester_stats = {}
        
    def __curves_update(self, time, time_check=True):
        
        if isinstance(time, datetime):
            time = time.timestamp()

        minute = int(time) // (CURVES_PLOT_INTERVAL_MINS*60)

        if time_check:
            if minute == self.last_curve_minute:
                return

        self.last_curve_minute = minute

        self.tester_curves["time"].append(time)
        self.tester_curves["balance"].append(self.AccountInfo.balance)
        self.tester_curves["equity"].append(self.AccountInfo.equity)
        self.tester_curves["margin"].append(self.AccountInfo.margin)
    
    def __account_state_update(self, account_info: AccountInfo):
        
        self.AccountInfo = account_info
        
    def account_info(self) -> AccountInfo:
        
        """Gets info on the current trading account."""
        
        if self.IS_TESTER:
            return self.AccountInfo
        
        mt5_ac_info = self.mt5_instance.account_info()
        if  mt5_ac_info is None:
            self.logger.warning(f"Failed to obtain MT5 account info, MT5 Error = {self.mt5_instance.last_error()}")
            return
            
        return mt5_ac_info
    
    def symbol_info(self, symbol: str) -> SymbolInfo:    
        
        """Gets data on the specified financial instrument."""

        if symbol not in self.symbol_info_cache:
            info = self.mt5_instance.symbol_info(symbol)
            if info is None:
                self.logger.warning(f"Failed to obtain symbol info for {symbol}")
                return None

            self.symbol_info_cache[symbol] = info

        return self.symbol_info_cache[symbol]

    def symbol_info_tick(self, symbol: str) -> Tick:
        """Get the last tick for the specified financial instrument.

        Returns:
            Tick: Returns the tick data as a named tuple Tick. Returns None in case of an error. The info on the error can be obtained using last_error().
        """

        tick = None
        if self.IS_TESTER:
            try:
                tick = self.tick_cache[symbol]
            except KeyError:
                self.logger.warning(f"{symbol} not found in the tick cache")
        else:
            try:
                tick = self.mt5_instance.symbol_info_tick(symbol)
            except Exception as e:
                self.logger.warning(f"Failed. MT5 Error = {self.mt5_instance.last_error()}")
            
        return tick
    
    def TickUpdate(self, symbol: str, tick: any):
        """Updates ticks in the StrategyTester/Simulator

        Args:
            symbol (str): An instrument for such a tick
            tick (any): Either a dictionary or a custom tick type named Tick from Tick
        """
        
        if not isinstance(tick, Tick) and not isinstance(tick, dict) and not isinstance(tick, tuple):
            self.logger.critical("Failed to update ticks, Invalid type received. It can either be a Tick, dict, or tuple datatypes")
            
        if isinstance(tick, dict):
            tick = make_tick_from_dict(tick)
        
        if isinstance(tick, tuple):
            tick = make_tick_from_tuple(tick)
        
        self.tick_cache[symbol] = tick

    @staticmethod
    def __mt5_data_to_dicts(rates) -> list[dict]:
        
        if rates is None or len(rates) == 0:
            return []

        # structured numpy array from MT5
        if rates.dtype.names is not None:
            return [
                {name: r[name].item() if hasattr(r[name], "item") else r[name]
                for name in rates.dtype.names}
                for r in rates
            ]

        raise TypeError(f"Unsupported rates format: {type(rates)}, dtype={rates.dtype}")

    def copy_rates_range(self, symbol: str, timeframe: int, date_from: datetime, date_to: datetime):
        """Get bars in the specified date range from the MetaTrader 5 terminal.

        Args:
            symbol (str): Financial instrument name, for example, "EURUSD". Required unnamed parameter.
            timeframe (int): Timeframe the bars are requested for. Set by a value from the TIMEFRAME enumeration. Required unnamed parameter.
            date_from (datetime): Date the bars are requested from. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. Bars with the open time >= date_from are returned. Required unnamed parameter.
            date_to (datetime): Date, up to which the bars are requested. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. Bars with the open time <= date_to are returned. Required unnamed parameter.

            Returns:
                Returns bars as the numpy array with the named time, open, high, low, close, tick_volume, spread and real_volume columns. Returns None in case of an error. The info on the error can be obtained using MetaTrader5.last_error().
        """

        date_from = ensure_utc(date_from)
        date_to = ensure_utc(date_to)

        if self.IS_TESTER:

            # instead of getting data from MetaTrader 5, get data stored in our custom directories

            path = os.path.join(self.history_dir, "Bars", symbol, TIMEFRAME2STRING_MAP[timeframe])
            os.makedirs(path, exist_ok=True)

            lf = pl.scan_parquet(path)

            try:
                rates = (
                    lf
                    .filter(
                            (pl.col("time") >= pl.lit(date_from)) &
                            (pl.col("time") <= pl.lit(date_to))
                        ) # get bars between date_from and date_to
                    .sort("time", descending=True)
                    .select([
                        pl.col("time").dt.epoch("s").cast(pl.Int64).alias("time"),

                        pl.col("open"),
                        pl.col("high"),
                        pl.col("low"),
                        pl.col("close"),
                        pl.col("tick_volume"),
                        pl.col("spread"),
                        pl.col("real_volume"),
                    ]) # return only what's required
                    .collect(engine="streaming") # the streaming engine, doesn't store data in memory
                ).to_dicts()

                rates = np.array(rates)[::-1] # reverse an array so it becomes oldest -> newest

            except Exception as e:
                self.logger.warning(f"Failed to copy rates from {date_from} to {date_to} {e}")
                return np.array(dict())
        else:

            rates = self.mt5_instance.copy_rates_range(symbol, timeframe, date_from, date_to)
            rates = np.array(self.__mt5_data_to_dicts(rates))

            if rates is None:
                self.logger.warning(f"Failed to copy rates. MetaTrader 5 error = {self.mt5_instance.last_error()}")
                return np.array(dict())

        return rates

    def copy_rates_from(self, symbol: str, timeframe: int, date_from: datetime, count: int) -> np.array:

        """Get bars from the MetaTrader 5 terminal starting from the specified date.

        Args:
            symbol: Financial instrument name, for example, "EURUSD". Required unnamed parameter.
            timeframe: Timeframe the bars are requested for. Set by a value from the TIMEFRAME enumeration. Required unnamed parameter.
            date_from: Date of opening of the first bar from the requested sample. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. Required unnamed parameter.

            count: Number of bars to receive. Required unnamed parameter.

        Returns:
            Returns bars as the numpy array with the named time, open, high, low, close, tick_volume, spread and real_volume columns. Return None in case of an error. The info on the error can be obtained using last_error().
        """

        date_from = ensure_utc(date_from)
        rates = np.array(dict())

        if self.IS_TESTER:

            # instead of getting data from MetaTrader 5, get data stored in our custom directories

            date_to = date_from + timedelta(seconds=PeriodSeconds(timeframe) * count)
            rates = self.copy_rates_range(symbol=symbol, timeframe=timeframe, date_from=date_from, date_to=date_to)

            if len(rates) == 0:
                self.logger.warning(f"Failed to to copy {count} bars from {date_from}")

        else:

            rates = self.mt5_instance.copy_rates_from(symbol, timeframe, date_from, count)
            rates = np.array(self.__mt5_data_to_dicts(rates))

            if rates is None:
                self.logger.warning(f"Failed to copy rates. MetaTrader 5 error = {self.mt5_instance.last_error()}")
                return np.array(dict())

        return rates

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int) -> np.array:

        """
        Get bars from the MetaTrader 5 terminal starting from the specified index.

        Parameters:
            symbol (str): Financial instrument name, for example, "EURUSD". Required unnamed parameter.
            timeframe (int): MT5 timeframe the bars are requested for.
            start_pos (int): Initial index of the bar the data are requested from. The numbering of bars goes from present to past. Thus, the zero bar means the current one. Required unnamed parameter.
            count (int): Number of bars to receive. Required unnamed parameter.

        Returns:
            Returns bars as the numpy array with the named time, open, high, low, close, tick_volume, spread and real_volume columns. Returns None in case of an error. The info on the error can be obtained using last_error().
        """

        tick = self.symbol_info_tick(symbol=symbol)

        if tick is None:
            self.logger.critical(
                "Time information not found in the ticker, call the function 'TickUpdate' giving it the latest tick information"
            )

            return None

        if self.IS_TESTER:

            now = tick.time

            date_from = now
            if isinstance(now, int) or isinstance(now, float):
                date_from = datetime.fromtimestamp(now)

            date_from += timedelta(seconds=PeriodSeconds(timeframe) * start_pos)

            rates = self.copy_rates_from(symbol=symbol,
                                         timeframe=timeframe,
                                         date_from=date_from,
                                         count=count)

            if len(rates) == 0:
                self.logger.warning(f"no rates found from {date_from} bars: count")
        else:

            rates = self.mt5_instance.copy_rates_from_pos(symbol, timeframe, start_pos, count)
            rates = np.array(self.__mt5_data_to_dicts(rates))

            if rates is None:
                self.logger.warning(f"Failed to copy rates. MetaTrader 5 error = {self.mt5_instance.last_error()}")
                return None

        return rates

    def __tick_flag_mask(self, flags: int) -> int:
        if flags == self.mt5_instance.COPY_TICKS_ALL:
            return (
                self.mt5_instance.TICK_FLAG_BID
                | self.mt5_instance.TICK_FLAG_ASK
                | self.mt5_instance.TICK_FLAG_LAST
                | self.mt5_instance.TICK_FLAG_VOLUME
                | self.mt5_instance.TICK_FLAG_BUY
                | self.mt5_instance.TICK_FLAG_SELL
            )

        mask = 0
        if flags & self.mt5_instance.COPY_TICKS_INFO:
            mask |= self.mt5_instance.TICK_FLAG_BID | self.mt5_instance.TICK_FLAG_ASK
        if flags & self.mt5_instance.COPY_TICKS_TRADE:
            mask |= self.mt5_instance.TICK_FLAG_LAST | self.mt5_instance.TICK_FLAG_VOLUME

        return mask

    def copy_ticks_from(self, symbol: str, date_from: datetime, count: int, flags: int=MetaTrader5.COPY_TICKS_ALL) -> np.array:
        
        """Get ticks from the MetaTrader 5 terminal starting from the specified date.

        Args:
            symbol(str): Financial instrument name, for example, "EURUSD". Required unnamed parameter.
            date_from(datetime): Date of opening of the first bar from the requested sample. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. Required unnamed parameter.

            count(int): Number of ticks to receive. Required unnamed parameter.
            flags(int): A flag to define the type of the requested ticks. COPY_TICKS_INFO – ticks with Bid and/or Ask changes, COPY_TICKS_TRADE – ticks with changes in Last and Volume, COPY_TICKS_ALL – all ticks. Flag values are described in the COPY_TICKS enumeration. Required unnamed parameter.

        Returns:
            Returns ticks as the numpy array with the named time, bid, ask, last and flags columns. The 'flags' value can be a combination of flags from the TICK_FLAG enumeration. Return None in case of an error. The info on the error can be obtained using last_error().
        """
        
        date_from = ensure_utc(date_from)
        flag_mask = self.__tick_flag_mask(flags)

        if self.IS_TESTER:    
            
            path = os.path.join(self.history_dir, "Ticks", symbol)
            os.makedirs(path, exist_ok=True)

            lf = pl.scan_parquet(path)

            try:
                ticks = (
                    lf
                    .filter(pl.col("time") >= pl.lit(date_from)) # get data starting at the given date
                    .filter((pl.col("flags") & flag_mask) != 0)
                    .sort(
                        ["time", "time_msc"],
                        descending=[False, False]
                    )
                    .limit(count) # limit the request to a specified number of ticks
                    .select([
                        pl.col("time").dt.epoch("s").cast(pl.Int64).alias("time"),

                        pl.col("bid"),
                        pl.col("ask"),
                        pl.col("last"),
                        pl.col("volume"),
                        pl.col("time_msc"),
                        pl.col("flags"),
                        pl.col("volume_real"),
                    ])
                    .collect(engine="streaming") # the streaming engine, doesn't store data in memory
                ).to_dicts()

                ticks = np.array(ticks)

            except Exception as e:
                self.logger.warning(f"Failed to copy ticks {e}")
                return None
        else:
            
            ticks = self.mt5_instance.copy_ticks_from(symbol, date_from, count, flags)
            ticks = np.array(self.__mt5_data_to_dicts(ticks))
            
            if ticks is None:
                self.logger.warning(f"Failed to copy ticks. MetaTrader 5 error = {self.mt5_instance.last_error()}")
                return None
            
        return ticks
    
    
    def copy_ticks_range(self, symbol: str, date_from: datetime, date_to: datetime, flags: int=MetaTrader5.COPY_TICKS_ALL) -> np.array:
        
        """Get ticks for the specified date range from the MetaTrader 5 terminal.

        Args:
            symbol(str): Financial instrument name, for example, "EURUSD". Required unnamed parameter.
            date_from(datetime): Date of opening of the first bar from the requested sample. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. Required unnamed parameter.

            date_to(datetime): Date, up to which the ticks are requested. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. Required unnamed parameter.
            flags(int): A flag to define the type of the requested ticks. COPY_TICKS_INFO – ticks with Bid and/or Ask changes, COPY_TICKS_TRADE – ticks with changes in Last and Volume, COPY_TICKS_ALL – all ticks. Flag values are described in the COPY_TICKS enumeration. Required unnamed parameter.

        Returns:
            Returns ticks as the numpy array with the named time, bid, ask, last and flags columns. The 'flags' value can be a combination of flags from the TICK_FLAG enumeration. Return None in case of an error. The info on the error can be obtained using last_error().
        """
        
        date_from = ensure_utc(date_from)
        date_to = ensure_utc(date_to)
        
        flag_mask = self.__tick_flag_mask(flags)

        if self.IS_TESTER:    
            
            path = os.path.join(self.history_dir, "Ticks", symbol)
            os.makedirs(path, exist_ok=True)
            
            lf = pl.scan_parquet(path)

            try:
                ticks = (
                    lf
                    .filter(
                            (pl.col("time") >= pl.lit(date_from)) &
                            (pl.col("time") <= pl.lit(date_to))
                        ) # get ticks between date_from and date_to
                    .filter((pl.col("flags") & flag_mask) != 0)
                    .sort(
                        ["time", "time_msc"],
                        descending=[False, False]
                    )
                    .select([
                        pl.col("time").dt.epoch("s").cast(pl.Int64).alias("time"),

                        pl.col("bid"),
                        pl.col("ask"),
                        pl.col("last"),
                        pl.col("volume"),
                        pl.col("time_msc"),
                        pl.col("flags"),
                        pl.col("volume_real"),
                    ]) 
                    .collect(engine="streaming") # the streaming engine, doesn't store data in memory
                ).to_dicts()

                ticks = np.array(ticks)
            
            except Exception as e:
                self.logger.warning(f"Failed to copy ticks {e}")
                return None
        else:
            
            ticks = self.mt5_instance.copy_ticks_range(symbol, date_from, date_to, flags)
            ticks = np.array(self.__mt5_data_to_dicts(ticks))
            
            if ticks is None:
                self.logger.warning(f"Failed to copy ticks. MetaTrader 5 error = {self.mt5_instance.last_error()}")
                return None
            
        return ticks
    

    def orders_total(self) -> int:
        
        """Get the number of active orders.
        
        Returns (int): The number of active orders in either a simulator or MetaTrader 5
        """
        
        if self.IS_TESTER:
            return len(self.__orders_container__)
        try:
            total = self.mt5_instance.orders_total()
        except Exception as e:
            self.logger.error(f"MetaTrader5 error = {e}")
            return -1
        
        return total
    
    def orders_get(self, symbol: Optional[str] = None, group: Optional[str] = None, ticket: Optional[int] = None) -> tuple[TradeOrder]:
                
        """Get active orders with the ability to filter by symbol or ticket. There are three call options.

        Args:
            symbol (str | optional): Symbol name. If a symbol is specified, the ticket parameter is ignored.
            group (str | optional): The filter for arranging a group of necessary symbols. If the group is specified, the function returns only active orders meeting a specified criteria for a symbol name.
            
            ticket (int | optional): Order ticket (ORDER_TICKET).
        
        Returns:
        
            list: Returns info in the form of a tuple structure (TradeOrder). Return None in case of an error. The info on the error can be obtained using last_error().
        """
        
        if self.IS_TESTER:
            
            orders = self.__orders_container__

            # no filters → return all orders
            if symbol is None and group is None and ticket is None:
                return tuple(orders)

            # symbol filter (highest priority)
            if symbol is not None:
                return tuple(o for o in orders if o.symbol == symbol)

            # group filter
            if group is not None:
                return tuple(o for o in orders if fnmatch.fnmatch(o.symbol, group))

            # ticket filter
            if ticket is not None:
                return tuple(o for o in orders if o.ticket == ticket)

            return tuple()
        
        try:
            if symbol is not None:
                return self.mt5_instance.orders_get(symbol=symbol)

            if group is not None:
                return self.mt5_instance.orders_get(group=group)

            if ticket is not None:
                return self.mt5_instance.orders_get(ticket=ticket)

            return self.mt5_instance.orders_get()

        except Exception as e:
            self.logger.error(f"MetaTrader5 error = {e}")
            return None

    def positions_total(self) -> int:
        """Get the number of open positions in MetaTrader 5 client.

        Returns:
            int: number of positions
        """
        
        if self.IS_TESTER:
            return len(self.__positions_container__)        
        try:
            total = self.mt5_instance.positions_total()
        except Exception as e:
            self.logger.error(f"MetaTrader5 error = {e}")
            return -1
        
        return total

    def positions_get(self, symbol: Optional[str] = None, group: Optional[str] = None, ticket: Optional[int] = None) -> tuple[TradePosition]:
        
        """Get open positions with the ability to filter by symbol or ticket. There are three call options.

        Args:
            symbol (str | optional): Symbol name. If a symbol is specified, the ticket parameter is ignored.
            group (str | optional): The filter for arranging a group of necessary symbols. Optional named parameter. If the group is specified, the function returns only positions meeting a specified criteria for a symbol name.
            
            ticket (int | optional): Position ticket -> https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer
        
        Returns:

            list: Returns info in the form of a tuple structure (TradePosition). Return None in case of an error. The info on the error can be obtained using last_error().
        """
        
        if self.IS_TESTER:
            
            positions = self.__positions_container__

            # no filters → return all positions
            if symbol is None and group is None and ticket is None:
                return tuple(positions)

            # symbol filter (highest priority)
            if symbol is not None:
                return tuple(o for o in positions if o.symbol == symbol)

            # group filter
            if group is not None:
                return tuple(o for o in positions if fnmatch.fnmatch(o.symbol, group))

            # ticket filter
            if ticket is not None:
                return tuple(o for o in positions if o.ticket == ticket)

            return tuple()
        
        try:
            if symbol is not None:
                return self.mt5_instance.positions_get(symbol=symbol)

            if group is not None:
                return self.mt5_instance.positions_get(group=group)

            if ticket is not None:
                return self.mt5_instance.positions_get(ticket=ticket)

            return self.mt5_instance.positions_get()

        except Exception as e:
            self.logger.error(f"MetaTrader5 error = {e}")
            return None

    def history_orders_total(self, date_from: datetime, date_to: datetime) -> int:
        
        # date range is a requirement
        
        if date_from is None or date_to is None:
            self.logger.error("date_from and date_to must be specified")
            return None
            
        date_from = ensure_utc(date_from)
        date_to = ensure_utc(date_to)
        
        if self.IS_TESTER:
        
            date_from_ts = int(date_from.timestamp())
            date_to_ts   = int(date_to.timestamp())
            
            return sum(
                        1
                        for o in self.__orders_history_container__
                        if date_from_ts <= o.time_setup <= date_to_ts
                    )

        try:
            total = self.mt5_instance.history_orders_total(date_from, date_to)
        except Exception as e:
            self.logger.error(f"MetaTrader5 error = {e}")
            return -1
        
        return total
    
    def history_orders_get(self, 
                           date_from: datetime,
                           date_to: datetime,
                           group: Optional[str] = None,
                           ticket: Optional[int] = None,
                           position: Optional[int] = None
                           ) -> tuple[TradeOrder]:
        
        if self.IS_TESTER:

            orders = self.__orders_history_container__

            # ticket filter (highest priority)
            if ticket is not None:
                return tuple(o for o in orders if o.ticket == ticket)

            # position filter
            if position is not None:
                return tuple(o for o in orders if o.position_id == position)

            # date range is a requirement  
            if date_from is None or date_to is None:
                self.logger.error("date_from and date_to must be specified")
                return None

            date_from_ts = int(ensure_utc(date_from).timestamp())
            date_to_ts   = int(ensure_utc(date_to).timestamp())

            filtered = (
                o for o in orders
                if date_from_ts <= o.time_setup <= date_to_ts
            ) # obtain orders that fall within this time range

            # optional group filter
            if group is not None:
                filtered = (
                    o for o in filtered
                    if fnmatch.fnmatch(o.symbol, group)
                )

            return tuple(filtered)
    
        try: # we are not on the strategy tester simulation
            
            if ticket is not None:
                return self.mt5_instance.history_orders_get(date_from, date_to, ticket=ticket)

            if position is not None:
                return self.mt5_instance.history_orders_get(date_from, date_to, position=position)

            if date_from is None or date_to is None:
                raise ValueError("date_from and date_to are required")

            date_from = ensure_utc(date_from)
            date_to   = ensure_utc(date_to)

            if group is not None:
                return self.mt5_instance.history_orders_get(
                    date_from, date_to, group=group
                )

            return self.mt5_instance.history_orders_get(date_from, date_to)

        except Exception as e:
            self.logger.error(f"MetaTrader5 error = {e}")
            return None
    
    def history_deals_total(self, date_from: datetime, date_to: datetime) -> int:
        """
        Get the number of deals in history within the specified date range.

        Args:
            date_from (datetime): Date the orders are requested from. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. 
            
            date_to (datetime, required): Date, up to which the orders are requested. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01.
        
        Returns:
            An integer value.
        """

        if date_from is None or date_to is None:
            self.logger.error("date_from and date_to must be specified")
            return -1

        date_from = ensure_utc(date_from)
        date_to   = ensure_utc(date_to)

        if self.IS_TESTER:

            date_from_ts = int(date_from.timestamp())
            date_to_ts   = int(date_to.timestamp())

            return sum(
                1
                for d in self.__deals_history_container__
                if date_from_ts <= d.time <= date_to_ts
            )

        try:
            return self.mt5_instance.history_deals_total(date_from, date_to)

        except Exception as e:
            self.logger.error(f"MetaTrader5 error = {e}")
            return -1
    
    def history_deals_get(self,
                          date_from: datetime,
                          date_to: datetime,
                          group: Optional[str] = None,
                          ticket: Optional[int] = None,
                          position: Optional[int] = None
                        ) -> tuple[TradeDeal]:
        """Gets deals from trading history within the specified interval with the ability to filter by ticket or position.

        Args:
            date_from (datetime): Date the orders are requested from. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. 
            
            date_to (datetime, required): Date, up to which the orders are requested. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01.
            
            group (str, optional):  The filter for arranging a group of necessary symbols. Optional named parameter. If the group is specified, the function returns only deals meeting a specified criteria for a symbol name.
            
            ticket (int, optional): Ticket of an order (stored in DEAL_ORDER) all deals should be received for. If not specified, the filter is not applied.
            
            position (int, optional): Ticket of a position (stored in DEAL_POSITION_ID) all deals should be received for. If not specified, the filter is not applied.

        Raises:
            ValueError: MetaTrader5 error

        Returns:
            tuple[TradeDeal]: information about deals
        """
                
        if self.IS_TESTER:

            deals = self.__deals_history_container__

            # ticket filter (highest priority)
            if ticket is not None:
                return tuple(d for d in deals if d.ticket == ticket)

            # position filter
            if position is not None:
                return tuple(d for d in deals if d.position_id == position)

            # date range is a requirement  
            if date_from is None or date_to is None:
                self.logger.error("date_from and date_to must be specified")
                return None

            date_from_ts = int(ensure_utc(date_from).timestamp())
            date_to_ts   = int(ensure_utc(date_to).timestamp())

            filtered = (
                d for d in deals
                if date_from_ts <= d.time <= date_to_ts
            ) # obtain orders that fall within this time range

            # optional group filter
            if group is not None:
                filtered = (
                    d for d in filtered
                    if fnmatch.fnmatch(d.symbol, group)
                )

            return tuple(filtered)
    
        try: # we are not on the strategy tester simulation
            
            if ticket is not None:
                return self.mt5_instance.history_deals_get(date_from, date_to, ticket=ticket)

            if position is not None:
                return self.mt5_instance.history_deals_get(date_from, date_to, position=position)

            if date_from is None or date_to is None:
                raise ValueError("date_from and date_to are required")

            date_from = ensure_utc(date_from)
            date_to   = ensure_utc(date_to)

            if group is not None:
                return self.mt5_instance.history_deals_get(
                    date_from, date_to, group=group
                )

            return self.mt5_instance.history_deals_get(date_from, date_to)

        except Exception as e:
            self.logger.error(f"MetaTrader5 error = {e}")
            return None

    def __generate_deal_ticket(self) -> int:
        return len(self.__deals_history_container__)+1

    def __generate_order_history_ticket(self) -> int:
        return len(self.__orders_history_container__)+1

    @staticmethod
    def __generate_order_ticket(ts: float) -> int:
        ts_i = int(ts * 1000)  # convert seconds -> ms integer
        rand = secrets.randbits(6)
        return (ts_i << 6) | rand

    @staticmethod
    def __generate_position_ticket(ts: float) -> int:
        ts_i = int(ts * 1000)
        rand = secrets.randbits(6)
        return (ts_i << 6) | rand

    def __calc_commission(self) -> float:
        """
        MT5-style commission calculation.
        """

        return -0.2
    
    def __position_to_order(self, position: TradePosition, ticket=None) -> TradeOrder:
        """
        Converts an opened position into a FILLED order
        (MT5 orders history behavior)
        """

        return TradeOrder(
            ticket=ticket if ticket is not None else position.ticket,
            time_setup=position.time,
            time_setup_msc=position.time_msc,
            time_done=position.time,
            time_done_msc=position.time_msc,
            time_expiration=0,

            type=position.type,
            type_time=self.mt5_instance.ORDER_TIME_GTC,
            type_filling=self.mt5_instance.ORDER_FILLING_FOK,
            state=self.mt5_instance.ORDER_STATE_FILLED,

            magic=position.magic,
            position_id=position.ticket,
            position_by_id=0,
            reason=position.reason,

            volume_initial=position.volume,
            volume_current=position.volume,

            price_open=position.price_open,
            sl=position.sl,
            tp=position.tp,
            price_current=position.price_current,
            price_stoplimit=0.0,

            symbol=position.symbol,
            comment=position.comment,
            external_id=position.external_id,
        )
    
    def __getTradeValidators(self, symbol: str):
        """Returns TradeValidators instance for the given symbol."""
        
        if symbol not in self.trade_validators_cache:
            
            self.trade_validators_cache[symbol] = TradeValidators(
                symbol_info=self.symbol_info(symbol),
                logger=self.logger,
                mt5_instance=self.mt5_instance
            )
            
        return self.trade_validators_cache[symbol]

    def order_send(self, request: dict):
        """
        Sends a request to perform a trading operation from the terminal to the trade server. The function is similar to OrderSend in MQL5.
        """

        # -----------------------------------------------------
        
        if not self.IS_TESTER:
            result = self.mt5_instance.order_send(request)
            if result is None or result.retcode != self.mt5_instance.TRADE_RETCODE_DONE:
                self.logger.warning(f"MT5 failed: {error_description.trade_server_return_code_description(self.mt5_instance.last_error()[0])}")
                return None
            return result
        
        # -------------------- Extract request -----------------------------
        
        action     = request.get("action")
        order_type = request.get("type", None)
        symbol     = request.get("symbol")
        volume     = float(request.get("volume", 0))
        price      = float(request.get("price", 0))
        sl         = float(request.get("sl", 0))
        tp         = float(request.get("tp", 0))
        ticket     = int(request.get("ticket", -1))
        
        # try:
        
        ticks_info = self.symbol_info_tick(symbol)
        symbol_info = self.symbol_info(symbol)
        ac_info = self.account_info()
        
        # except Exception as e:
        #     self.logger.critical(f"Failed to obtain necessary ticks, symbol, or account info: {e}")
        #     return None
        
        now = ticks_info.time
        ts = now.timestamp() if isinstance(now, datetime) else now

        msc = int(ts * 1000)
        
        if order_type is not None:
            if order_type not in ORDER_TYPE_MAP.keys():
                self.logger.critical("Invalid order type")
                return None 
    
        trade_validators = self.__getTradeValidators(symbol=symbol)

        # ------------------ MARKET DEAL (open or close) ------------------
        
        if action == self.mt5_instance.TRADE_ACTION_DEAL:
            
            def deal_reason_gen() -> int:
                eps = pow(10, -symbol_info.digits)
                if TradeValidators.price_equal(a=price, b=sl, eps=eps):
                    return self.mt5_instance.DEAL_REASON_SL
                
                if TradeValidators.price_equal(a=price, b=tp, eps=eps):
                    return self.mt5_instance.DEAL_REASON_TP
                
                return self.mt5_instance.DEAL_REASON_EXPERT
                
            # ---------- CLOSE POSITION ----------
            
            position_ticket = request.get("position", -1)
            if position_ticket != -1:
                pos = next(
                    (p for p in self.__positions_container__ if p.ticket == position_ticket),
                    None,
                )
                
                if not pos:
                    return {"retcode": self.mt5_instance.TRADE_RETCODE_INVALID}

                # validate position close request
                
                if pos.type == order_type:
                    self.logger.critical("Failed to close an order. Order type must be the opposite")
                    return None
                
                if order_type == self.mt5_instance.ORDER_TYPE_BUY: # For a sell order/position
                    
                    if not TradeValidators.price_equal(a=price, b=ticks_info.ask, eps=pow(10, -symbol_info.digits)):
                        self.logger.critical(f"Failed to close ORDER_TYPE_SELL. Price {price} is not equal to bid {ticks_info.bid}")
                        return None
                        
                elif order_type == self.mt5_instance.ORDER_TYPE_SELL: # For a buy order/position
                    if not TradeValidators.price_equal(a=price, b=ticks_info.bid, eps=pow(10, -symbol_info.digits)):
                        self.logger.critical(f"Failed to close ORDER_TYPE_BUY. Price {price} is not equal to bid {ticks_info.bid}")
                        return None
                
                # update account info    
                
                commission = self.__calc_commission() # calculate commission
                
                new_balance = self.AccountInfo.balance + pos.profit + commission
                ac_margin = self.AccountInfo.margin
                new_margin = ac_margin - pos.margin
                
                self.AccountInfo = self.AccountInfo._replace(
                    balance=new_balance,
                    margin=new_margin,
                )

                self.__account_monitoring(pos_must_exist=False)
                self.__curves_update(now, time_check=False)
                self.__positions_container__.remove(pos) 
                
                # self.__orders_history_container__.append(
                #     self.__position_to_order(position=position) #TODO:
                # )
                
                deal_ticket = self.__generate_deal_ticket()
                self.__deals_history_container__.append(
                    TradeDeal(
                        ticket=deal_ticket,
                        order=0,
                        time=ts,
                        time_msc=msc,
                        type=order_type,
                        entry=self.mt5_instance.DEAL_ENTRY_OUT,
                        magic=request.get("magic", 0),
                        position_id=pos.ticket,
                        reason=deal_reason_gen(),
                        volume=volume,
                        price=price,
                        commission=commission,
                        swap=0,
                        profit=pos.profit,
                        fee=0,
                        symbol=symbol,
                        comment=request.get("comment", ""),
                        external_id="",
                        balance=self.AccountInfo.balance,
                    )
                )

                self.logger.info(f"Position: {position_ticket} closed!")
                self.logger.debug(f"balance: {self.AccountInfo.balance:.2f} equity: {self.AccountInfo.equity:2f} pl: {self.AccountInfo.profit:.2f}")
                
                return {
                    "retcode": self.mt5_instance.TRADE_RETCODE_DONE,
                    "deal": deal_ticket,
                }
                
            # ---------- OPEN POSITION ----------
            
            if not trade_validators.is_valid_entry_price(order_type=order_type, price=price, tick_info=ticks_info):
                return None
            
            # validate new stops 
            
            if not trade_validators.is_valid_sl(entry=price, sl=sl, order_type=order_type):
                return None
            if not trade_validators.is_valid_tp(entry=price, tp=tp, order_type=order_type):
                return None
            
            # validate the lotsize
            
            if not trade_validators.is_valid_lotsize(lotsize=volume):
                return None
            
            total_volume = sum([pos.volume for pos in self.__positions_container__]) + sum([order.volume_current for order in self.__orders_container__])
            if trade_validators.is_symbol_volume_reached(symbol_volume=total_volume, volume_limit=symbol_info.volume_limit):
                return None
            
            margin_required = self.order_calc_margin(order_type=order_type, symbol=symbol, volume=volume, price=price)

            if not trade_validators.is_there_enough_money(margin_required=margin_required, free_margin=ac_info.margin_free):
                return None
            
            position_ticket = StrategyTester.__generate_position_ticket(ts=ts)
            order_ticket    = StrategyTester.__generate_order_ticket(ts=ts)
            deal_ticket     = self.__generate_deal_ticket()

            position = TradePosition(
                ticket=position_ticket,
                time=ts,
                time_msc=msc,
                time_update=ts,
                time_update_msc=msc,
                type=order_type,
                magic=request.get("magic", 0),
                identifier=position_ticket,
                reason=self.mt5_instance.DEAL_REASON_EXPERT,
                volume=volume,
                price_open=price,
                sl=sl,
                tp=tp,
                price_current=price,
                swap=0,
                profit=0,
                symbol=symbol,
                comment=request.get("comment", ""),
                external_id="",
                margin=margin_required
            )
            
            self.__positions_container__.append(position)

            self.__orders_history_container__.append(
                self.__position_to_order(position=position, ticket=self.__generate_order_history_ticket())
            )
            
            self.__deals_history_container__.append(
                TradeDeal(
                    ticket=deal_ticket,
                    order=order_ticket,
                    time=ts,
                    time_msc=msc,
                    type=order_type,
                    entry=self.mt5_instance.DEAL_ENTRY_IN,
                    magic=request.get("magic", 0),
                    position_id=position_ticket,
                    reason=deal_reason_gen(),
                    volume=volume,
                    price=price,
                    commission=self.__calc_commission(),
                    swap=0,
                    profit=0,
                    fee=0,
                    symbol=symbol,
                    comment=request.get("comment", ""),
                    external_id="",
                    balance=self.AccountInfo.balance,
                )
            )
            
            
            self.logger.info(f"Position: {position_ticket} opened!")
                
            return {
                "retcode": self.mt5_instance.TRADE_RETCODE_DONE,
                "deal": deal_ticket,
                "order": order_ticket,
                "position": position_ticket,
            }
            
        # --------------------- PENDING order --------------------------
        
        elif action == self.mt5_instance.TRADE_ACTION_PENDING:
            
            if trade_validators.is_max_orders_reached(open_orders=len(self.__orders_container__), 
                                                      ac_limit_orders=ac_info.limit_orders):
                return None
            
            if not trade_validators.is_valid_sl(entry=price, sl=sl, order_type=order_type) or not trade_validators.is_valid_tp(entry=price, tp=tp, order_type=order_type):
                return None
            
            total_volume = sum([pos.volume for pos in self.__positions_container__]) + sum([order.volume_current for order in self.__orders_container__])
            if trade_validators.is_symbol_volume_reached(symbol_volume=total_volume, volume_limit=symbol_info.volume_limit):
                return None
            
            order_ticket = self.__generate_order_ticket()

            order = TradeOrder(
                    ticket=order_ticket,
                    time_setup=ts,
                    time_setup_msc=msc,
                    time_done=0,
                    time_done_msc=0,
                    time_expiration=request.get("expiration", 0),
                    type=order_type,
                    type_time=request.get("type_time", 0),
                    type_filling=request.get("type_filling", 0),
                    state=self.mt5_instance.ORDER_STATE_PLACED,
                    magic=request.get("magic", 0),
                    position_id=0,
                    position_by_id=0,
                    reason=self.mt5_instance.DEAL_REASON_EXPERT,
                    volume_initial=volume,
                    volume_current=volume,
                    price_open=price,
                    sl=sl,
                    tp=tp,
                    price_current=price,
                    price_stoplimit=request.get("price_stoplimit", 0),
                    symbol=symbol,
                    comment=request.get("comment", ""),
                    external_id="",
                )
                
            self.__orders_container__.append(order) 
            self.__orders_history_container__.append(order)
            
            self.logger.info(f"Pending order: {order_ticket} placed!")
                
            return {
                "retcode": self.mt5_instance.TRADE_RETCODE_DONE,
                "order": order_ticket,
            }
            
        elif action == self.mt5_instance.TRADE_ACTION_SLTP:
            
            position_ticket = request.get("position", -1)

            pos = next((p for p in self.__positions_container__ if p.ticket == position_ticket), None)
            if not pos:
                return {"retcode": self.mt5_instance.TRADE_RETCODE_INVALID}

            # --- Correct reference prices ---
            entry_price = pos.price_open
            market_price = ticks_info.bid if pos.type == self.mt5_instance.POSITION_TYPE_BUY else ticks_info.ask

            # --- Validate SL / TP relative to ENTRY ---
            if sl > 0:
                if not trade_validators.is_valid_sl(entry=entry_price, sl=sl, order_type=pos.type):
                    return None

            if tp > 0:
                if not trade_validators.is_valid_tp(entry=entry_price, tp=tp, order_type=pos.type):
                    return None

            # --- Validate freeze level against MARKET ---
            if sl > 0:
                if not trade_validators.is_valid_freeze_level(tick_info=ticks_info, entry=market_price, stop_price=sl, order_type=pos.type):
                    return None

            if tp > 0:
                if not trade_validators.is_valid_freeze_level(tick_info=ticks_info, entry=market_price, stop_price=tp, order_type=pos.type):
                    return None

            # --- APPLY MODIFICATION ---
            idx = self.__positions_container__.index(pos)

            updated_pos = pos._replace(
                sl=sl,
                tp=tp,
                time_update=ts,
                time_update_msc=msc
            )

            self.__positions_container__[idx] = updated_pos

            self.logger.info(f"Position: {ticket} Modified!")
                
            return {"retcode": self.mt5_instance.TRADE_RETCODE_DONE}
        
            
        # -------------------- REMOVE pending order ------------------------
        
        elif action == self.mt5_instance.TRADE_ACTION_MODIFY: # Modifying pending orders

            order_ticket = request.get("order", -1)

            order = next(
                (o for o in self.__orders_container__ if o.ticket == order_ticket),
                None,
            )

            if not order:
                return {"retcode": self.mt5_instance.TRADE_RETCODE_INVALID}

            # validate new stops 
            
            if not trade_validators.is_valid_freeze_level(entry=price, stop_price=sl, order_type=order_type):
                return None
            if not trade_validators.is_valid_freeze_level(entry=price, stop_price=tp, order_type=order_type):
                return None
                
            # Modify ONLY allowed fields
            
            idx = self.__orders_container__.index(order)

            updated_order = order._replace(
                price_open=price,
                sl=sl,
                tp=tp,
                time_expiration = request.get("expiration", order.time_expiration),
                price_stoplimit = request.get("price_stoplimit", order.price_stoplimit)
            )

            self.__orders_container__[idx] = updated_order
            
            self.logger.info(f"Pending Order: {ticket} Modified!")
                
            return {"retcode": self.mt5_instance.TRADE_RETCODE_DONE}
        
        elif action == self.mt5_instance.TRADE_ACTION_REMOVE:
            
            ticket = request.get("order", -1)
            
            self.__orders_container__ = [
                o for o in self.__orders_container__ if o.ticket != ticket
            ]
            
            self.logger.info(f"Pending order: {ticket} removed!")
                
            return {"retcode": self.mt5_instance.TRADE_RETCODE_DONE}

        return {
            "retcode": self.mt5_instance.TRADE_RETCODE_INVALID,
            "comment": "Unsupported trade action",
        }
    
    def order_calc_profit(self, 
                        order_type: int,
                        symbol: str,
                        volume: float,
                        price_open: float,
                        price_close: float) -> float:
        """
        Return profit in the account currency for a specified trading operation.
        
        Args:
            order_type (int): The type of position taken, either 0 (buy) or 1 (sell).
            symbol (str): Financial instrument name. 
            volume (float):   Trading operation volume.
            price_open (float): Open Price.
            price_close (float): Close Price.
        """
        
        sym = self.symbol_info(symbol)
        
        if self.IS_TESTER:
            
            contract_size = sym.trade_contract_size

            direction = 0

            # --- Determine direction ---
            if order_type in BUY_ACTIONS:
                direction = 1
            elif order_type in SELL_ACTIONS:
                direction = -1

            # --- Core profit calculation ---

            calc_mode = sym.trade_calc_mode
            price_delta = (price_close - price_open) * direction

            try:
                # ------------------ FOREX / CFD / STOCKS -----------------------
                if calc_mode in (
                    self.mt5_instance.SYMBOL_CALC_MODE_FOREX,
                    self.mt5_instance.SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE,
                    self.mt5_instance.SYMBOL_CALC_MODE_CFD,
                    self.mt5_instance.SYMBOL_CALC_MODE_CFDINDEX,
                    self.mt5_instance.SYMBOL_CALC_MODE_CFDLEVERAGE,
                    self.mt5_instance.SYMBOL_CALC_MODE_EXCH_STOCKS,
                    self.mt5_instance.SYMBOL_CALC_MODE_EXCH_STOCKS_MOEX,
                ):
                    profit = price_delta * contract_size * volume

                # ---------------- FUTURES --------------------
                elif calc_mode in (
                    self.mt5_instance.SYMBOL_CALC_MODE_FUTURES,
                    self.mt5_instance.SYMBOL_CALC_MODE_EXCH_FUTURES,
                    # self.mt5_instance.SYMBOL_CALC_MODE_EXCH_FUTURES_FORTS,
                ):
                    tick_value = sym.trade_tick_value
                    tick_size = sym.trade_tick_size

                    if tick_size <= 0:
                        self.logger.critical("Invalid tick size")
                        return 0.0

                    profit = price_delta * volume * (tick_value / tick_size)

                # ---------- BONDS -------------------
                
                elif calc_mode in (
                    self.mt5_instance.SYMBOL_CALC_MODE_EXCH_BONDS,
                    self.mt5_instance.SYMBOL_CALC_MODE_EXCH_BONDS_MOEX,
                ):
                    face_value = sym.trade_face_value
                    accrued_interest = sym.trade_accrued_interest

                    profit = (
                        volume
                        * contract_size
                        * (price_close * face_value + accrued_interest)
                        - volume
                        * contract_size
                        * (price_open * face_value)
                    )

                # ------ COLLATERAL -------
                elif calc_mode == self.mt5_instance.SYMBOL_CALC_MODE_SERV_COLLATERAL:
                    liquidity_rate = sym.trade_liquidity_rate
                    market_price = (
                        self.tick_cache[symbol].ask if order_type == self.mt5_instance.ORDER_TYPE_BUY else self.tick_cache[symbol].bid
                    )

                    profit = (
                        volume
                        * contract_size
                        * market_price
                        * liquidity_rate
                    )

                else:
                    self.logger.critical(
                        f"Unsupported trade calc mode: {calc_mode}"
                    )
                    return 0.0

                return round(profit, 2)
                
            except Exception as e:
                self.logger.critical(f"Failed: {e}")
                return 0.0
            
        # if we are not on the strategy tester
            
        try:
            profit = self.mt5_instance.order_calc_profit(
                order_type,
                symbol,
                volume,
                price_open,
                price_close
            )
        
        except Exception as e:
            self.logger.critical(f"Failed to calculate profit of a position, MT5 error = {self.mt5_instance.last_error()}")
            return np.nan
        
        return profit
    
    def order_calc_margin(self, order_type: int, symbol: str, volume: float, price: float) -> float:
        """
        Return margin in the account currency to perform a specified trading operation.
        
        """

        if volume <= 0 or price <= 0:
            self.logger.error("order_calc_margin failed: invalid volume or price")
            return 0.0

        if not self.IS_TESTER:
            try:
                return round(self.mt5_instance.order_calc_margin(order_type, symbol, volume, price), 2)
            except Exception:
                self.logger.warning(f"Failed: MT5 Error = {self.mt5_instance.last_error()}")
                return 0.0

        # IS_TESTER = True
        sym = self.symbol_info(symbol)

        contract_size = sym.trade_contract_size
        leverage = max(self.account_info().leverage, 1)

        margin_rate = (
            sym.margin_initial
            if sym.margin_initial > 0
            else sym.margin_maintenance
        )
        
        if margin_rate <= 0: # if margin rate is zero set it to 1
            margin_rate = 1.0

        mode = sym.trade_calc_mode

        if mode == self.mt5_instance.SYMBOL_CALC_MODE_FOREX:
            margin = (volume * contract_size * price) / leverage

        elif mode == self.mt5_instance.SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE:
            margin = volume * contract_size * price

        elif mode in (
            self.mt5_instance.SYMBOL_CALC_MODE_CFD,
            self.mt5_instance.SYMBOL_CALC_MODE_CFDINDEX,
            self.mt5_instance.SYMBOL_CALC_MODE_EXCH_STOCKS,
            self.mt5_instance.SYMBOL_CALC_MODE_EXCH_STOCKS_MOEX,
        ):
            margin = volume * contract_size * price * margin_rate

        elif mode == self.mt5_instance.SYMBOL_CALC_MODE_CFDLEVERAGE:
            margin = (volume * contract_size * price * margin_rate) / leverage

        elif mode in (
            self.mt5_instance.SYMBOL_CALC_MODE_FUTURES,
            self.mt5_instance.SYMBOL_CALC_MODE_EXCH_FUTURES,
            # self.mt5_instance.SYMBOL_CALC_MODE_EXCH_FUTURES_FORTS,
        ):
            margin = volume * sym.margin_initial

        elif mode in (
            self.mt5_instance.SYMBOL_CALC_MODE_EXCH_BONDS,
            self.mt5_instance.SYMBOL_CALC_MODE_EXCH_BONDS_MOEX,
        ):
            margin = (
                volume
                * contract_size
                * sym.trade_face_value
                * price
                / 100
            )

        elif mode == self.mt5_instance.SYMBOL_CALC_MODE_SERV_COLLATERAL:
            margin = 0.0

        else:
            self.logger.warning(f"Unknown calc mode {mode}, fallback margin formula used")
            margin = (volume * contract_size * price) / leverage

        return round(margin, 2)
    
    def __positions_monitoring(self):
        """
        Monitors all open positions and updates the account:
        - updates profit
        - checks SL / TP
        - closes positions when hit
        """

        positions_found = len(self.__positions_container__)

        self.positions_total_margin = 0
        self.positions_unrealized_pl = 0

        for i in range(positions_found- 1, -1, -1):
            
            pos = self.__positions_container__[i]
            
            tick = self.symbol_info_tick(pos.symbol)

            # --- Determine close price and opposite order type ---
            if pos.type == self.mt5_instance.POSITION_TYPE_BUY:
                price = tick.bid
                close_type = self.mt5_instance.ORDER_TYPE_SELL
            elif pos.type == self.mt5_instance.POSITION_TYPE_SELL:
                price = tick.ask
                close_type = self.mt5_instance.ORDER_TYPE_BUY
            else:
                self.logger.warning("Unknown position type")
                continue

            # --- Update floating profit ---
            
            profit = self.order_calc_profit(
                    order_type=pos.type,
                    symbol=pos.symbol,
                    volume=pos.volume,
                    price_open=pos.price_open,
                    price_close=price
                )
            
            self.positions_unrealized_pl += profit
            self.positions_total_margin += pos.margin
            
            # --- Check SL / TP ---
            hit_tp = False
            hit_sl = False

            if pos.tp > 0:
                hit_tp = (
                    price >= pos.tp if pos.type == self.mt5_instance.POSITION_TYPE_BUY
                    else price <= pos.tp
                )

            if pos.sl > 0:
                hit_sl = (
                    price <= pos.sl if pos.type == self.mt5_instance.POSITION_TYPE_BUY
                    else price >= pos.sl
                )
                
            pos = pos._replace(
                profit=profit,
                price_current=price,
                time_update=tick.time,
                time_update_msc=tick.time_msc
            )

            # MUST write it back
            self.__positions_container__[i] = pos

            # self.logger.debug(pos) #TODO:
            
            if not (hit_tp or hit_sl):
                continue

            # --- Close position ---
            request = {
                "action": self.mt5_instance.TRADE_ACTION_DEAL,
                "type": close_type,
                "symbol": pos.symbol,
                "price": price,
                "volume": pos.volume,
                "position": pos.ticket,
                "comment": "TP hit" if hit_tp else "SL hit",
            }

            self.order_send(request)

    def __account_monitoring(self, pos_must_exist: bool = True):

        # ------- monitor the account only if there is at least one position ------

        if (len(self.__positions_container__) > 0) if pos_must_exist else True:

            new_equity = self.AccountInfo.balance + self.positions_unrealized_pl
            self.AccountInfo = self.AccountInfo._replace(
                profit=self.positions_unrealized_pl,
                equity=new_equity,
                margin=self.positions_total_margin,
                margin_free = new_equity - self.positions_total_margin,
                margin_level = new_equity / self.positions_total_margin * 100 if self.positions_total_margin > 0 else 0
            )

        
    def __pending_orders_monitoring(self):
        
        """
        Monitors pending orders:
        - handles expiration
        - triggers STOP / LIMIT orders correctly
        - converts them into market positions
        """

        for order in list(self.__orders_container__):

            symbol = order.symbol
            tick = self.tick_cache[symbol]

            # --- Expiration handling ---
            if order.time_expiration > 0 and tick.time >= order.time_expiration:
                self.__orders_container__.remove(order)
                continue

            triggered = False
            deal_type = None
            deal_price = None

            # -------- BUY ORDERS --------
            if order.type == self.mt5_instance.ORDER_TYPE_BUY_LIMIT:
                if tick.ask <= order.price:
                    triggered = True
                    deal_type = self.mt5_instance.ORDER_TYPE_BUY
                    deal_price = order.price

            elif order.type == self.mt5_instance.ORDER_TYPE_BUY_STOP:
                if tick.ask >= order.price:
                    triggered = True
                    deal_type = self.mt5_instance.ORDER_TYPE_BUY
                    deal_price = tick.ask

            elif order.type == self.mt5_instance.ORDER_TYPE_BUY_STOP_LIMIT:
                if tick.ask >= order.price:
                    # Convert to BUY LIMIT at stoplimit price
                    order.type = self.mt5_instance.ORDER_TYPE_BUY_LIMIT
                    order.price = order.price_stoplimit
                continue

            # -------- SELL ORDERS --------
            elif order.type == self.mt5_instance.ORDER_TYPE_SELL_LIMIT:
                if tick.bid >= order.price:
                    triggered = True
                    deal_type = self.mt5_instance.ORDER_TYPE_SELL
                    deal_price = order.price

            elif order.type == self.mt5_instance.ORDER_TYPE_SELL_STOP:
                if tick.bid <= order.price:
                    triggered = True
                    deal_type = self.mt5_instance.ORDER_TYPE_SELL
                    deal_price = tick.bid

            elif order.type == self.mt5_instance.ORDER_TYPE_SELL_STOP_LIMIT:
                if tick.bid <= order.price:
                    order.type = self.mt5_instance.ORDER_TYPE_SELL_LIMIT
                    order.price = order.price_stoplimit
                continue

            if not triggered:
                continue

            # ----- Execute pending order -----
            request = {
                "action": self.mt5_instance.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "type": deal_type,
                "price": deal_price,
                "sl": order.sl,
                "tp": order.tp,
                "volume": order.volume_current,
                "magic": order.magic,
                "comment": order.comment,
            }

            result = self.order_send(request)

            # ----- Remove pending order after successful execution -----
            if result and result.get("retcode") == self.mt5_instance.TRADE_RETCODE_DONE:
                self.__orders_container__.remove(order)
    
    def _bar_to_tick(self, symbol, bar):
        """
            Creates a synthetic tick from a bar (MT5-style).
            Uses OPEN price.
        """

        price = bar["open"] if isinstance(bar, dict) else bar[1]
        time = bar["time"] if isinstance(bar, dict) else bar[0]
        spread = bar["spread"] if isinstance(bar, dict) else bar[6]
        tv = bar["tick_volume"] if isinstance(bar, dict) else bar[5]
        
        return {
            
            "time": time,
            "bid": price,
            "ask": price + spread * self.symbol_info(symbol).point,
            "last": price,
            "volume": tv,
            "time_msc": time.timestamp(),
            "flags": 0,
            "volume_real": 0,
        }
    
    def OnTick(self, ontick_func):
        """Calls the assigned function upon the receival of new tick(s)

        Args:
            ontick_func (_type_): A function to be called on every tick
        """

        self.__TesterInit()
        
        modelling = self.tester_config["modelling"]
        if modelling == "real_ticks" or modelling == "every_tick":
            
            total_ticks = sum(ticks_info["size"] for ticks_info in self.TESTER_ALL_TICKS_INFO)
            self.tester_stats["Ticks"] = total_ticks

            self.logger.debug(f"total number of ticks: {total_ticks}")

            with tqdm(total=total_ticks, desc="StrategyTester Progress", unit="tick") as pbar:
                while True:

                    self.__positions_monitoring()
                    self.__account_monitoring()
                    self.__pending_orders_monitoring()
                    
                    any_tick_processed = False

                    for ticks_info in self.TESTER_ALL_TICKS_INFO:

                        symbol = ticks_info["symbol"]
                        size = ticks_info["size"]
                        counter = ticks_info["counter"]

                        if counter >= size:
                            continue

                        current_tick = ticks_info["ticks"].row(counter)
                        
                        current_tick = make_tick_from_tuple(current_tick)
                        self.TickUpdate(symbol=symbol, tick=current_tick)
                        
                        self.__curves_update(current_tick.time)
                        ontick_func()

                        ticks_info["counter"] = counter + 1
                        any_tick_processed = True

                        pbar.update(1)

                    if not any_tick_processed:
                        break
                    
        elif modelling == "new_bar" or modelling == "1-minute-ohlc":
            
            bars_ = [bars_info["size"] for bars_info in self.TESTER_ALL_BARS_INFO]
            total_bars = sum(bars_)
            self.tester_stats["Ticks"] = total_bars
            
            self.logger.debug(f"total number of bars: {total_bars}")

            with tqdm(total=total_bars, desc="StrategyTester Progress", unit="bar") as pbar:
                while True:

                    self.__positions_monitoring()
                    self.__account_monitoring()
                    self.__pending_orders_monitoring()
                    
                    any_bar_processed = False

                    for bars_info in self.TESTER_ALL_BARS_INFO:

                        symbol = bars_info["symbol"]
                        size = bars_info["size"]
                        counter = bars_info["counter"]

                        if counter >= size:
                            continue

                        current_tick = self._bar_to_tick(symbol=symbol, bar=bars_info["bars"].row(counter))

                        self.__curves_update(current_tick["time"])
                        
                        # Getting ticks at the current bar
                        
                        self.TickUpdate(symbol=symbol, tick=current_tick)
                        ontick_func()

                        bars_info["counter"] = counter + 1
                        any_bar_processed = True

                        pbar.update(1)

                    if not any_bar_processed:
                        break
        
        self.__TesterDeinit()

    def __ontick_symbol(self, symbol: str, modelling: str, ontick_func: any):

        info = None
        is_tick_mode = False

        if modelling in ("new_bar", "1-minute-ohlc"):
            info = next(
                (x for x in self.TESTER_ALL_BARS_INFO if x["symbol"] == symbol),
                None,
            )

        if modelling in ("real_ticks", "every_tick"):
            info = next(
                (x for x in self.TESTER_ALL_TICKS_INFO if x["symbol"] == symbol),
                None,
            )

            is_tick_mode = True

        if info is None:
            return

        ticks = info["ticks"] if is_tick_mode else info["bars"]
        size = info["size"]
        counter = 0

        self.logger.info(f"{symbol} total number of ticks: {size}")

        with tqdm(total=size, desc=f"StrategyTester Progress on {symbol}", unit="tick" if is_tick_mode else "bar") as pbar:
            while counter < size:

                tick = None
                if is_tick_mode:
                    tick = ticks.row(counter)
                else:
                    tick = self._bar_to_tick(symbol=symbol, bar=ticks.row(counter)) # a bar=tick is not actually a tick, rather a bar

                if tick is None:
                    pbar.update(1)
                    continue

                self.TickUpdate(symbol=symbol, tick=tick)
                ontick_func()

                positions_found = len(self.positions_get(symbol=symbol))>0
                pending_orders_found = len(self.orders_get(symbol=symbol))>0

                if positions_found: # we monitor the account and positions only if they exist

                    self.__positions_n_account_monitoring()

                    self.__curves_update(tick.time)

                if pending_orders_found:
                    self.__pending_orders_monitoring()

                counter += 1
                pbar.update(1)

    def ParallelOnTick(self, ontick_func):
        """Calls the assigned function upon the receival of new tick(s)

        Args:
            ontick_func (_type_): A function to be called on every tick
        """

        self.__TesterInit()

        symbols = self.tester_config["symbols"]
        modelling = self.tester_config["modelling"]
        max_workers = len(symbols)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futs = {executor.submit(self.__ontick_symbol, s, modelling, ontick_func): s for s in symbols}

        self.__TesterDeinit()

    def __make_balance_deal(self, time: datetime) -> TradeDeal:

        time_sec = int(time.timestamp())
        time_msc = int(time.timestamp() * 1000)

        return TradeDeal(
            ticket=self.__generate_deal_ticket(),
            order=0,
            time=time_sec,
            time_msc=time_msc,
            type=self.mt5_instance.DEAL_TYPE_BALANCE,
            entry=self.mt5_instance.DEAL_ENTRY_IN,
            magic=0,
            position_id=0,
            reason=np.nan,
            volume=np.nan,
            price=np.nan,
            commission=0.0,
            swap=0.0,
            profit=0.0,
            fee=0.0,
            symbol="",
            balance=self.AccountInfo.balance,
            comment="",
            external_id=""
        )

    def __TesterInit(self):
        
        self.__deals_history_container__.append(
            self.__make_balance_deal(time=self.tester_config["start_date"])
        )
        
    def __TesterDeinit(self):
        
        # self.__deals_history_container__.append(
        #     self.__make_balance_deal(time=self.tester_config["end_date"])
        # )

        self.tester_stats["Symbols"] = len(self.tester_config["symbols"])

        profits = []
        losses = []
        total_trades = 0
        
        max_consec_win_count = 0
        max_consec_win_money = 0.0

        max_consec_loss_count = 0
        max_consec_loss_money = 0.0

        max_profit_streak_money = 0.0
        max_profit_streak_count = 0

        max_loss_streak_money = 0.0
        max_loss_streak_count = 0

        cur_win_count = 0
        cur_win_money = 0.0

        cur_loss_count = 0
        cur_loss_money = 0.0

        win_streaks = []
        loss_streaks = []
        
        short_trades_won = 0
        long_trades_won = 0
        
        total_short_trades = 0
        total_long_trades = 0
        
        for deal in self.__deals_history_container__:
            if deal.entry == self.mt5_instance.DEAL_ENTRY_OUT: # a closed position
                
                total_trades +=1
                if deal.type == self.mt5_instance.DEAL_TYPE_BUY:
                    total_long_trades += 1
                
                if deal.type == self.mt5_instance.DEAL_TYPE_SELL:
                    total_short_trades += 1
                
                profit = deal.profit
                    
                if profit > 0: # A win
                    
                    profits.append(profit)
                    
                    # reset loss streak
                    if cur_loss_count > 0:
                        loss_streaks.append(cur_loss_count)
                        cur_loss_count = 0
                        cur_loss_money = 0.0

                    cur_win_count += 1
                    cur_win_money += profit

                    # longest win streak
                    if cur_win_count > max_consec_win_count:
                        max_consec_win_count = cur_win_count
                        max_consec_win_money = cur_win_money

                    # most profitable win streak
                    if cur_win_money > max_profit_streak_money:
                        max_profit_streak_money = cur_win_money
                        max_profit_streak_count = cur_win_count

                    if deal.type == self.mt5_instance.DEAL_TYPE_BUY:
                        long_trades_won += 1
                    
                    if deal.type == self.mt5_instance.DEAL_TYPE_SELL:
                        short_trades_won += 1
                        
                else: # A loss
                    
                    losses.append(profit)
                    
                    # reset win streak
                    if cur_win_count > 0:
                        win_streaks.append(cur_win_count)
                        cur_win_count = 0
                        cur_win_money = 0.0

                    cur_loss_count += 1
                    cur_loss_money += profit

                    # longest loss streak
                    if cur_loss_count > max_consec_loss_count:
                        max_consec_loss_count = cur_loss_count
                        max_consec_loss_money = cur_loss_money

                    # largest losing streak
                    if cur_loss_money < max_loss_streak_money:
                        max_loss_streak_money = cur_loss_money
                        max_loss_streak_count = cur_loss_count
                    
        
        self.tester_stats["Gross Profit"] = np.sum(profits) if profits else 0
        self.tester_stats["Gross Loss"] = np.sum(losses) if losses else 0
        self.tester_stats["Net Profit"] = self.tester_stats["Gross Profit"] + self.tester_stats["Gross Loss"]
        
        self.tester_stats["Profit Factor"] = self.tester_stats["Gross Profit"] / (self.tester_stats["Gross Loss"]+epsilon)
        
        self.tester_stats["Expected Payoff"] = (
            self.tester_stats["Net Profit"] / total_trades
            if total_trades > 0 else 0
        )

        def max_drawdown(curve):
            try:
                peak = curve[0]
            except IndexError:
                peak = 0

            max_dd = 0.0

            for value in curve:
                peak = max(peak, value)
                dd = peak - value
                max_dd = max(max_dd, dd)

            return max_dd

        returns = np.diff(self.tester_curves["equity"])

        sharpe = (
            np.mean(returns) / np.std(returns)
            if len(returns) > 1 and np.std(returns) > 0 else 0.0
        )
        
        self.tester_stats["Sharpe Ratio"] = sharpe
        
        self.tester_stats["Equity Drawdown Absolute"] = max_drawdown(self.tester_curves["equity"])
        self.tester_stats["Balance Drawdown Absolute"] = max_drawdown(self.tester_curves["balance"])
        
        self.tester_stats["Recovery Factor"] = (
            self.tester_stats["Net Profit"] / max(self.tester_stats["Balance Drawdown Absolute"], 1)
        )

        self.tester_stats["Equity Drawdown Relative"] = (
            self.tester_stats["Equity Drawdown Absolute"] / max(self.tester_curves["equity"]) * 100
            if self.tester_curves["equity"] else 0.0
        )
        
        self.tester_stats["Balance Drawdown Relative"] = (
            self.tester_stats["Balance Drawdown Absolute"] / max(self.tester_curves["balance"]) * 100
            if self.tester_curves["balance"] else 0.0
        )
        
        self.tester_stats["Balance Drawdown Maximal"] = max_drawdown(self.tester_curves["balance"])
        self.tester_stats["Equity Drawdown Maximal"] = max_drawdown(self.tester_curves["equity"])
        
        self.tester_stats["Total Trades"] = total_trades
        self.tester_stats["Total Deals"] = len(self.__deals_history_container__)
        
        self.tester_stats["Total Short Trades"] = total_short_trades
        self.tester_stats["Total Long Trades"] = total_long_trades
        
        self.tester_stats["Short Trades Won"] = short_trades_won
        self.tester_stats["Long Trades Won"] = long_trades_won
        
        self.tester_stats["Profit Trades"] = len(profits) if profits else 0
        self.tester_stats["Loss Trades"] = len(losses) if losses else 0
        
        self.tester_stats["Largest Profit Trade"] = np.max(profits) if profits else 0
        self.tester_stats["Largest Loss Trade"] = np.min(losses) if losses else 0
        
        self.tester_stats["Average Profit Trade"] = np.mean(profits) if profits else 0
        self.tester_stats["Average Loss Trade"] = np.mean(losses) if losses else 0
        
        self.tester_stats["Maximum Consecutive Wins"] = max_profit_streak_count
        self.tester_stats["Maximum Consecutive Losses"] = max_loss_streak_count
        
        self.tester_stats["Maximum Consecutive Wins Money"] = max_profit_streak_money
        self.tester_stats["Maximum Consecutive Losses Money"] = max_loss_streak_money
        
        self.tester_stats["Average Consecutive Wins"] = np.mean(win_streaks)
        self.tester_stats["Average Consecutive Losses"] = np.mean(loss_streaks)
        
        # AHPR / GHPR
        
        self.tester_stats["AHPR"] = np.prod(1 + returns) ** (1/len(returns)) if len(returns) else 0
        self.tester_stats["GHPR"] = np.prod(1 + returns) if len(returns) else 0

        # generate a report at the end
        
        self.__GenerateTesterReport(output_file=f"Reports/{self.tester_config['bot_name']}-report.html")
    
    def _plot_tester_curves(self, output_path: str) -> str:
        
        curves = self.tester_curves

        if not curves["time"]:
            return None

        # Convert timestamps → datetime
        times = [
            datetime.fromtimestamp(t) if isinstance(t, (int, float)) else t
            for t in curves["time"]
        ]

        plt.style.use("seaborn-v0_8-darkgrid")
        plt.figure(figsize=(10, 4))

        plt.plot(times, curves["balance"], label="Balance", linewidth=2, color="#1f77b4")
        plt.plot(times, curves["equity"], label="Equity", linewidth=1.5, color="#0b6623", alpha=0.6)

        # plt.plot(times, curves["margin"], label="Margin", linewidth=1, alpha=0.6)

        plt.legend(loc="lower left")
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, transparent=False)
        plt.close()

        return output_path

    def __GenerateTesterReport(self, output_file="StrategyTester report.html"):
        
        def render_order_rows(orders):
            rows = []

            for o in orders:
                rows.append(f"""
                <tr>
                    <td>{datetime.fromtimestamp(o.time_setup)}</td>
                    <td>{o.ticket}</td>
                    <td>{o.symbol}</td>
                    <td>{ORDER_TYPE_MAP.get(o.type, o.type)}</td>
                    <td class="text-end">{o.volume_initial:.2f} / {o.volume_current:.2f}</td>
                    <td class="text-end">{o.price_open:.5f}</td>
                    <td class="text-end">{"" if o.sl == 0 else f"{o.sl:.5f}"}</td>
                    <td class="text-end">{"" if o.tp == 0 else f"{o.tp:.5f}"}</td>
                    <td>{datetime.fromtimestamp(o.time_done) if o.time_done else ""}</td>
                    <td>{ORDER_STATE_MAP.get(o.state, o.state)}</td>
                    <td>{o.comment}</td>
                </tr>
                """)

            return "\n".join(rows)

        def render_deal_rows(deals):
            rows = []

            for d in deals:

                rows.append(f"""
                <tr>
                    <td>{datetime.fromtimestamp(d.time)}</td>
                    <td>{d.ticket}</td>
                    <td>{d.symbol}</td>
                    <td>{DEAL_TYPE_MAP[d.type]}</td>
                    <td>{DEAL_ENTRY_MAP[d.entry]}</td>
                    <td class="text-end">{d.volume:.2f}</td>
                    <td class="text-end">{d.price:.5f}</td>
                    <td class="text-end">{d.commission:.2f}</td>
                    <td class="text-end">{d.swap:.2f}</td>
                    <td class="text-end">{d.profit:.2f}</td>
                    <td>{d.comment}</td>
                    <td>{round(d.balance, 2)}</td>
                </tr>
                """)

            return "\n".join(rows)
        # ---------------------- write stats table to a template --------------------
        
        short_trades_won = self.tester_stats.get("Short Trades Won", 0)
        long_trades_won = self.tester_stats.get('Long Trades Won', 0)
        
        max_profit_streak_count = self.tester_stats["Maximum Consecutive Wins"]
        max_loss_streak_count = self.tester_stats["Maximum Consecutive Losses"]
        
        max_profit_streak_money = self.tester_stats["Maximum Consecutive Wins Money"]
        max_loss_streak_money = self.tester_stats["Maximum Consecutive Losses Money"]
        
        stats_table = f"""
            <table class="report-table table-sm table-striped">
                <tbody>
                    <tr>
                        <th>Initial Deposit</th><td class="number">{self.tester_config.get('deposit', 0)}</td>
                        <th>Ticks</th><td class="number">{self.tester_stats.get('Ticks', 0)}</td>
                        <th>Symbols</th><td class="number">{self.tester_stats.get('Symbols', 0)}</td>
                    </tr>
                    <tr>
                        <th>Total Net Profit</th><td class="number">{self.tester_stats.get('Net Profit', 0):.2f}</td>
                        <th>Balance Drawdown Absolute</th><td class="number">{self.tester_stats.get('Balance Drawdown Absolute', 0):.2f}</td>
                        <th>Equity Drawdown Absolute</th><td class="number">{self.tester_stats.get('Equity Drawdown Absolute', 0):.2f}</td>
                    </tr>
                    <tr>
                        <th>Gross Profit</th><td class="number">{self.tester_stats.get('Gross Profit', 0):.2f}</td>
                        <th>Balance Drawdown Maximal</th><td class="number">{self.tester_stats.get('Balance Drawdown Maximal', 0):.2f}</td>
                        <th>Equity Drawdown Maximal</th><td class="number">{self.tester_stats.get('Equity Drawdown Maximal', 0):.2f}</td>
                    </tr>
                    <tr>
                        <th>Gross Loss</th><td class="number">{self.tester_stats.get('Gross Loss', 0):.2f}</td>
                        <th>Balance Drawdown Relative</th><td class="number">{self.tester_stats.get('Balance Drawdown Relative', 0):.2f}%</td>
                        <th>Equity Drawdown Relative</th><td class="number">{self.tester_stats.get('Equity Drawdown Relative', 0):.2f}%</td>
                    </tr>
                    <tr>
                        <th>Profit Factor</th><td class="number">{self.tester_stats.get('Profit Factor', 0):.2f}</td>
                        <th>Expected Payoff</th><td class="number">{self.tester_stats.get('Expected Payoff', 0):.2f}</td>
                        <th>Margin Level</th><td class="number">{self.tester_stats.get('Margin Level', 0):.2f}%</td>
                    </tr>
                    <tr>
                        <th>Recovery Factor</th><td class="number">{self.tester_stats.get('Recovery Factor', 0):.2f}</td>
                        <th>Sharpe Ratio</th><td class="number">{self.tester_stats.get('Sharpe Ratio', 0):.2f}</td>
                        <th>Z-Score</th><td class="number">{self.tester_stats.get('Z-Score', 0):.2f}</td>
                    </tr>
                    <tr>
                        <th>AHPR</th><td class="number">{self.tester_stats.get('AHPR', 0):.4f}</td>
                        <th>LR Correlation</th><td class="number">{self.tester_stats.get('LR Correlation', 0):.2f}</td>
                        <th>OnTester result</th><td class="number">{self.tester_stats.get('OnTester result', 0)}</td>
                    </tr>
                    <tr>
                        <th>GHPR</th><td class="number">{self.tester_stats.get('GHPR', 0):.4f}</td>
                        <th>LR Standard Error</th><td class="number">{self.tester_stats.get('LR Standard Error', 0):.2f}</td>
                        <td></td><td></td>
                    </tr>
                    <tr>
                        <th>Total Trades</th><td class="number">{self.tester_stats.get('Total Trades', 0)}</td>
                        <th>Short Trades (won %)</th><td class="number">{short_trades_won} ({100*short_trades_won/(self.tester_stats.get('Total Short Trades',1)+epsilon):.2f}%)</td>
                        <th>Long Trades (won %)</th><td class="number">{long_trades_won} ({100*long_trades_won/(self.tester_stats.get('Total Long Trades',1)+epsilon):.2f}%)</td>
                    </tr>
                    <tr>
                        <th>Total Deals</th><td class="number">{self.tester_stats.get('Total Deals', 0)}</td>
                        <th>Profit Trades (% of total)</th><td class="number">{self.tester_stats.get('Profit Trades', 0)} ({100*self.tester_stats.get('Profit Trades',0)/max(self.tester_stats.get('Total Trades',1),1):.2f}%)</td>
                        <th>Loss Trades (% of total)</th><td class="number">{self.tester_stats.get('Loss Trades', 0)} ({100*self.tester_stats.get('Loss Trades',0)/max(self.tester_stats.get('Total Trades',1),1):.2f}%)</td>
                    </tr>
                    <tr>
                        <th>Largest Profit Trade</th><td class="number">{self.tester_stats.get('Largest Profit Trade', 0):.2f}</td>
                        <th>Largest Loss Trade</th><td class="number">{self.tester_stats.get('Largest Loss Trade', 0):.2f}</td>
                        <td></td><td></td>
                    </tr>
                    <tr>
                        <th>Average Profit Trade</th><td class="number">{self.tester_stats.get('Average Profit Trade', 0):.2f}</td>
                        <th>Average Loss Trade</th><td class="number">{self.tester_stats.get('Average Loss Trade', 0):.2f}</td>
                        <td></td><td></td>
                    </tr>
                    <tr>
                        <th>Max Consecutive Wins ($)</th><td class="number">{max_profit_streak_count} ({max_profit_streak_money:.2f})</td>
                        <th>Max Consecutive Losses ($)</th><td class="number">{max_loss_streak_count} ({max_loss_streak_money:.2f})</td>
                        <td></td><td></td>
                    </tr>
                    <tr>
                        <th>Max Consecutive Profit (count)</th><td class="number">{max_profit_streak_count} ({max_profit_streak_money:.2f})</td>
                        <th>Max Consecutive Loss (count)</th><td class="number">{max_loss_streak_count} ({max_loss_streak_money:.2f})</td>
                        <td></td><td></td>
                    </tr>
                    <tr>
                        <th>Average Consecutive Wins</th><td class="number">{self.tester_stats.get('Average Consecutive Wins', 0):.2f}</td>
                        <th>Average Consecutive Losses</th><td class="number">{self.tester_stats.get('Average Consecutive Losses', 0):.2f}</td>
                        <td></td><td></td>
                    </tr>
                </tbody>
            </table>
            """

        
        # ----------------------- append a PNG for curves to the HTML -----------------
        
        path = os.path.join(self.reports_dir, "images")
        os.makedirs(path, exist_ok=True)

        try:
            curve_img = self._plot_tester_curves(output_path=os.path.join(path, f"{self.tester_config['bot_name'].replace(' ', '_')}_curve.png"))
            curve_img = curve_img.replace(self.reports_dir+'\\', "")
        except Exception as e:
            self.logger.warning(f"Failed to generate a balance curve {e}")

        # ------------------ render orders and deals ------------------------------
        
        order_rows_html = render_order_rows(self.__orders_history_container__)
        deal_rows_html = render_deal_rows(self.__deals_history_container__)
        
        template = html_report_template()
        
        # we populate table's body
        
        html = (
            template
            .replace("{{STATS_TABLE}}", stats_table)
            .replace("{{ORDER_ROWS}}", order_rows_html)
            .replace("{{DEAL_ROWS}}", deal_rows_html)
            .replace(
            "{{CURVE_IMAGE}}",
            f'<img src="{curve_img}" class="img-fluid curve-img">' if curve_img else ""
            )
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)

        self.logger.info(f"Deals report saved to: {output_file}")
