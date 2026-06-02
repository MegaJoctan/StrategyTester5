from __future__ import annotations

from . import *
from typing import Optional, Literal, Union, Any
from strategytester5.MetaTrader5.constants import MetaTrader5Constants
import MetaTrader5
import logging
from .error_description import return_code_description
from .trade_validators import TradeValidators
from . import data
import fnmatch
from datetime import datetime
from functools import wraps

from ..MQL5.functions import PeriodSeconds


def requires_initialization(func):
    """
    A decorator for checking if virtual MT5 is initialized.

    Raises:
        A runtime error if a method is called before initialize().
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):

        if not self._initialized:
            raise RuntimeError(
                "VirtualMetaTrader5 is not initialized. "
                "Call initialize() first."
            )

        return func(self, *args, **kwargs)

    return wrapper

class VirtualMetaTrader5(MetaTrader5Constants):
    """The simulated MetaTrader5 Instance similar to [https://pypi.org/project/metatrader5/](https://pypi.org/project/metatrader5/)"""

    def __init__(self):
        """
        Instantiates & initializes the simulated MetaTrader5 instance
        """
        
        super().__init__()

        self.IS_OPTIMIZATION_MODE = False
        self.logger = None
        self.broker_data_path = ""
        self.history_manager = None

        self._last_error = None

        self._current_time: int = 0
        self._current_time_msc: int = -1

        # ----------------- MetaTrader5-Like Containers-----------------

        self.parent_mt5: MetaTrader5 = Any
        self.ACCOUNT: AccountInfo = Any
        self.TERMINAL_INFO: TerminalInfo = Any
        self.SYMBOL_INFO_CACHE = {}

        self.TICK_CACHE: dict[str, Tick] = {}

        self.ORDERS = []
        self.ORDERS_HISTORY = []
        self.POSITIONS = []
        self.DEALS = []
        self.TRADE_VALIDATORS_CACHE = {}

        # tickets

        self._positions_counter = 0
        self._orders_counter = 0

        self._initialized = False

    def initialize(self,
                   parent_mt5: MetaTrader5,
                   is_optimization_mode: bool,
                   custom_broker_data_path: str = "",
                   logger: Optional[logging.Logger] = None
                   ) -> bool:
        """
        Initializes the virtual MetaTrader5.

        Args:
            parent_mt5 (Any): MetaTrader5 API/client instance used for obtaining crucial information from the broker as an attempt to mimic the terminal.
            is_optimization_mode (bool): Whether to run the virtual terminal in optimization mode.
            custom_broker_data_path (bool | optional): Where custom folders for history are kept.
            logger (Logger | optional): Logger object for logging.

        Returns:
            True if successful and False otherwise.
        """

        # store global variables

        self.IS_OPTIMIZATION_MODE = is_optimization_mode
        self.parent_mt5 = parent_mt5
        self.logger = logger

        # checking the parent MetaTrader5

        if parent_mt5 is None:
            self.critical_log("Invalid parent_mt5 was received")
            return False

        # we export account information and all instrument info
        all_s_info = parent_mt5.symbols_get()

        if all_s_info is None:
            self.critical_log(f"Failed to obtain symbol info from the live MetaTrader5 instance, error = {parent_mt5.last_error()}")
            return False

        self.SYMBOL_INFO_CACHE = {s.name: s for s in all_s_info}

        ac_info = parent_mt5.account_info()

        if ac_info is None:
            self.critical_log(f"Failed to obtain account info from the live MetaTrader5 instance, error = {parent_mt5.last_error()}")
            return False

        self.ACCOUNT = AccountInfo(*ac_info)

        # broker's data

        self.broker_data_path = self.ACCOUNT.server if custom_broker_data_path == "" else custom_broker_data_path
        self.history_manager = data.HistoryManager(mt5_instance=parent_mt5, broker_data_path=self.broker_data_path)

        terminal_info = parent_mt5.terminal_info()
        if terminal_info is None:
            self.critical_log(f"Failed to obtain terminal information from a live MetaTrader5 instance, error = {parent_mt5.last_error()}")
            return False

        self.TERMINAL_INFO = terminal_info

        self._initialized = True
        return True

    @requires_initialization
    def reset_state(self):
        """
        Resets the internal state of the virtual MetaTrader5 object.
        """
        self._last_error = None

        self._current_time: int = 0
        self._current_time_msc: int = -1

        self._positions_counter = 0
        self._orders_counter = 0

        self.SYMBOL_INFO_CACHE = {}

        self.TICK_CACHE: dict[str, Tick] = {}

        # clear history

        self.ORDERS = []
        self.ORDERS_HISTORY = []
        self.POSITIONS = []
        self.DEALS = []
        self.TRADE_VALIDATORS_CACHE = {}

    def info_log(self, msg: str):
        if self.IS_OPTIMIZATION_MODE:
            return

        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg, stacklevel=3)

    def debug_log(self, msg: str):
        if self.IS_OPTIMIZATION_MODE:
            return

        if self.logger is None:
            print(msg)
        else:
            self.logger.debug(msg, stacklevel=3)

    def warning_log(self, msg: str):
        if self.IS_OPTIMIZATION_MODE:
            return

        if self.logger is None:
            print(msg)
        else:
            self.logger.warning(msg, stacklevel=3)

    def critical_log(self, msg: str):
        if self.IS_OPTIMIZATION_MODE:
            return

        if self.logger is None:
            print(msg)
        else:
            self.logger.critical(msg, stacklevel=3)

    def error_log(self, msg: str):
        if self.IS_OPTIMIZATION_MODE:
            return

        if self.logger is None:
            print(msg)
        else:
            self.logger.error(msg, stacklevel=3)

    @requires_initialization
    def current_time(self) -> int:
        """Returns the current time in seconds since 1970.01.01 00:00:00, as obtained from the latest tick update."""
        return self._current_time

    @requires_initialization
    def current_time_msc(self) -> int:
        """Returns the current time in milliseconds since 1970.01.01 00:00:00, as obtained from the latest tick update."""
        return self._current_time_msc

    @requires_initialization
    def _generate_order_history_ticket(self) -> int:
        return len(self.ORDERS_HISTORY) + 1

    @requires_initialization
    def _generate_deal_ticket(self) -> int:
        return len(self.DEALS) + 1

    @requires_initialization
    def _generate_order_ticket(self) -> int:
        self._orders_counter += 1
        return self._orders_counter

    @requires_initialization
    def _generate_position_ticket(self) -> int:
        self._positions_counter += 1
        return self._positions_counter

    @requires_initialization
    def last_error(self):
        """returns the last error from the terminal or the strategy tester"""
        return self._last_error

    @requires_initialization
    def account_info(self) -> Optional[AccountInfo]:
        """Gets info on the current trading account.

        [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5accountinfo_py)

        Returns:
            Trading account's information in a namedtuple (tuple) called AccountInfo
        """

        return self.ACCOUNT

    @requires_initialization
    def symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Gets data on the specified financial instrument.

        [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5symbolinfo_py)

        Returns:
            Symbol's information in a namedtuple (tuple) called SymbolInfo
        """

        if symbol not in self.SYMBOL_INFO_CACHE:
            self.warning_log(f"Failed to obtain symbol info for {symbol}")
            return None

        return self.SYMBOL_INFO_CACHE[symbol]

    @requires_initialization
    def symbol_info_tick(self, symbol: str) -> Tick:
        """Gets the last tick for the specified financial instrument.

        [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5symbolinfotick_py)

        Returns:
            Tick: Returns the tick data as a named tuple Tick. Returns None in case of an error. The info on the error can be obtained using last_error().
        """

        tick = None
        try:
            tick = self.TICK_CACHE[symbol]
        except KeyError:
            self.warning_log(f"{symbol} not found in the tick cache")

        return tick

    @requires_initialization
    def tick_update(self, symbol: str, tick: Union[Tick, dict, TICKS_DTYPE]):
        """
        Assigns the tick object to a virtual MetaTrader5 instance.

        Args:
            symbol: An instrument a given tick belongs to.
            tick: The tick object to be assigned.
        """

        if isinstance(tick, dict):
            tick = Tick(
                time=tick["time"],
                bid=tick["bid"],
                ask=tick["ask"],
                last=tick["last"],
                volume=tick["volume"],
                time_msc=tick["time_msc"],
                flags=tick["flags"],
                volume_real=tick["volume_real"],
            )

        elif isinstance(tick, np.void):
            tick = Tick(
                time=tick[0],
                bid=tick[1],
                ask=tick[2],
                last=tick[3],
                volume=tick[4],
                time_msc=tick[5],
                flags=tick[6],
                volume_real=tick[7],
            )

        elif hasattr(tick, "time") and hasattr(tick, "bid"):

            tick = Tick(
                time=tick.time,
                bid=tick.bid,
                ask=tick.ask,
                last=tick.last,
                volume=tick.volume,
                time_msc=tick.time_msc,
                flags=tick.flags,
                volume_real=tick.volume_real,
            )

        else:
            log = f"Unknown tick type {type(tick)}"
            self.critical_log(log)
            raise RuntimeError(log)

        self._current_time = tick.time
        self._current_time_msc = tick.time_msc
        self.TICK_CACHE[symbol] = tick

    @requires_initialization
    def copy_rates_range(self,
                         symbol: str,
                         timeframe: int,
                         date_from: datetime,
                         date_to: datetime,
                         parent_mt5_source: bool=False,
                         polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto"
                         ) -> Optional[
        RATES_DTYPE]:
        """Get bars in the specified date range from the MetaTrader 5 terminal.

        [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py)

        Args:
            symbol (str): Financial instrument name, for example, "EURUSD". Required unnamed parameter.
            timeframe (int): Timeframe the bars are requested for. Set by a value from the TIMEFRAME enumeration. Required unnamed parameter.
            date_from (datetime): Date the bars are requested from. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. Bars with the open time >= date_from are returned. Required unnamed parameter.
            date_to (datetime): Date, up to which the bars are requested. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. Bars with the open time <= date_to are returned. Required unnamed parameter.
            parent_mt5_source (bool): Whether to obtain rates directly from the `parent_mt5` (directly from the terminal) when set to True. Or, from custom broker's path created by the StrategyTester object.

            polars_collect_engine (str): Engine used by Polars when collecting rates from custom broker's path. Supported values are:
                - ``"auto"`` (default): Use Polars’ standard in-memory engine and
                    respect the ``POLARS_ENGINE_AFFINITY`` environment variable if set.
                - ``"in-memory"``: Explicitly use the default in-memory engine,
                    optimized with multi-threading and SIMD over Arrow data.
                - ``"streaming"``: Process queries in batches, enabling
                    larger-than-RAM datasets.
                - ``"gpu"``: Use NVIDIA GPUs via RAPIDS cuDF for accelerated execution.
                    Requires installing Polars with GPU support, e.g.:
                    ``pip install polars[gpu] --extra-index-url=https://pypi.nvidia.com``.

            Returns:
                Returns bars as the numpy array with the named time, open, high, low, close, tick_volume, spread and real_volume columns. Returns None in case of an error. The info on the error can be obtained using MetaTrader5.last_error().

            Notes:
                - In some cases, copying data directly from the terminal becomes cheap compared to reading from parquet files that introduce `file IO` operations that are computationally expensive. In such cases, `parent_mt5` becomes handy.
        """

        if not isinstance(date_from, datetime) or not isinstance(date_to, datetime):
            self.warning_log("Failed, both `date_from` and `date_to` must be datetime objects")
            return None

        if parent_mt5_source:
            rates = self.parent_mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
            self._last_error = self.parent_mt5.last_error()

        else:
            rates = self.history_manager.copy_rates_range_from_parquet(symbol, timeframe, date_from, date_to,
                                                                       polars_collect_engine=polars_collect_engine,
                                                                       broker_data_dir=self.broker_data_path,
                                                                       logger=self.logger)

        if rates is None or len(rates) == 0:
            self.warning_log(f"no rates found on {symbol} from {date_from} bars: {date_to}")
            return None

        return rates

    @requires_initialization
    def copy_rates_from(self,
                        symbol: str,
                        timeframe: int,
                        date_from: datetime,
                        count: int,
                        parent_mt5_source: bool = False,
                        polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto"
                        ) -> Optional[RATES_DTYPE]:

        """Get bars from the MetaTrader 5 terminal starting from the specified date.

        [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrom_py)

        Args:
            symbol (str): Financial instrument name, for example, "EURUSD". Required unnamed parameter.
            timeframe (int): Timeframe the bars are requested for. Set by a value from the TIMEFRAME enumeration. Required unnamed parameter.
            date_from (datetime): Date of opening of the first bar from the requested sample. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. Required unnamed parameter.
            count (int): Number of bars to receive. Required unnamed parameter.
            parent_mt5_source (bool): Whether to obtain rates directly from the `parent_mt5` (directly from the terminal) when set to True. Or, from custom broker's path created by the StrategyTester object.

            polars_collect_engine (str): Engine used by Polars when collecting rates from custom broker's path. Supported values are:
                - ``"auto"`` (default): Use Polars’ standard in-memory engine and
                    respect the ``POLARS_ENGINE_AFFINITY`` environment variable if set.
                - ``"in-memory"``: Explicitly use the default in-memory engine,
                    optimized with multi-threading and SIMD over Arrow data.
                - ``"streaming"``: Process queries in batches, enabling
                    larger-than-RAM datasets.
                - ``"gpu"``: Use NVIDIA GPUs via RAPIDS cuDF for accelerated execution.
                    Requires installing Polars with GPU support, e.g.:
                    ``pip install polars[gpu] --extra-index-url=https://pypi.nvidia.com``.

        Returns:
            Returns bars as the numpy array with the named time, open, high, low, close, tick_volume, spread and real_volume columns. Return None in case of an error. The info on the error can be obtained using last_error().

        Notes:
            - In some cases, copying data directly from the terminal becomes cheap compared to reading from parquet files that introduce `file IO` operations that are computationally expensive. In such cases, `parent_mt5` becomes handy.
        """

        if isinstance(date_from, (int, float)):
            date_from = datetime.fromtimestamp(date_from)

        # instead of getting data from MetaTrader 5, get data stored in our custom directories

        if parent_mt5_source:
            rates = self.parent_mt5.copy_rates_from(symbol, timeframe, date_from, count)
        else:

            date_to = self.current_time() - PeriodSeconds(timeframe) * count
            rates = self.history_manager.copy_rates_from_parquet(symbol,
                                                                 timeframe,
                                                                 date_from=date_from,
                                                                 history_start_date=date_to,
                                                                 count=count,
                                                                 broker_data_dir=self.broker_data_path,
                                                                 logger=self.logger,
                                                                 polars_collect_engine=polars_collect_engine)

        if rates is None or len(rates) == 0:
            self.warning_log(f"no rates found for {symbol} from {date_from} bars: {count}")
            return None

        return rates

    @requires_initialization
    def copy_rates_from_pos(self,
                            symbol: str,
                            timeframe: int,
                            start_pos: int,
                            count: int,
                            parent_mt5_source: bool=False,
                            polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto"
                            ) -> Optional[TICKS_DTYPE]:
        """
        Get bars from the MetaTrader 5 terminal starting from the specified index.

        [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrompos_py)

        Args:
            symbol (str): Financial instrument name, for example, "EURUSD". Required unnamed parameter.
            timeframe (int): MT5 timeframe the bars are requested for.
            start_pos (int): Initial index of the bar the data are requested from. The numbering of bars goes from present to past. Thus, the zero bar means the current one. Required unnamed parameter.
            count (int): Number of bars to receive. Required unnamed parameter.
            parent_mt5_source (bool): Whether to obtain rates directly from the `parent_mt5` (directly from the terminal) when set to True. Or, from custom broker's path created by the StrategyTester object.

            polars_collect_engine (str): Engine used by Polars when collecting rates from custom broker's path. Supported values are:
                - ``"auto"`` (default): Use Polars’ standard in-memory engine and
                    respect the ``POLARS_ENGINE_AFFINITY`` environment variable if set.
                - ``"in-memory"``: Explicitly use the default in-memory engine,
                    optimized with multi-threading and SIMD over Arrow data.
                - ``"streaming"``: Process queries in batches, enabling
                    larger-than-RAM datasets.
                - ``"gpu"``: Use NVIDIA GPUs via RAPIDS cuDF for accelerated execution.
                    Requires installing Polars with GPU support, e.g.:
                    ``pip install polars[gpu] --extra-index-url=https://pypi.nvidia.com``.

        Returns:
            Returns bars as the numpy array with the named time, open, high, low, close, tick_volume, spread and real_volume columns. Returns None in case of an error. The info on the error can be obtained using last_error().

        Notes:
            - In some cases, copying data directly from the terminal becomes cheap compared to reading from parquet files that introduce `file IO` operations that are computationally expensive. In such cases, `parent_mt5` becomes handy.
        """

        tick = self.symbol_info_tick(symbol=symbol)

        if not tick:
            self.critical_log(
                f"Time information not found in the ticker for {symbol}, call the function 'tick_update' giving it the latest tick information")
            return None

        if parent_mt5_source:
            rates = self.parent_mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
            self._last_error = self.parent_mt5.last_error()

        else:

            date_from = self.current_time() - PeriodSeconds(timeframe) * start_pos
            date_to = date_from - PeriodSeconds(timeframe) * count

            rates = self.history_manager.copy_rates_from_parquet(symbol,
                                                                 timeframe,
                                                                 date_from=date_from,
                                                                 history_start_date=date_to,
                                                                 count=count,
                                                                 broker_data_dir=self.broker_data_path,
                                                                 logger=self.logger,
                                                                 polars_collect_engine=polars_collect_engine)

        if rates is None or len(rates) == 0:
            self.debug_log(f"no rates found for {symbol} from {start_pos} bars: {count}")
            return None

        return rates

    def symbol_select(self, symbol: str, select: bool=False) -> bool:
        return True

    @requires_initialization
    def copy_ticks_range(self,
                         symbol: str,
                         date_from: datetime,
                         date_to: datetime,
                         flags: int = MetaTrader5.COPY_TICKS_ALL,
                         parent_mt5_source: bool=False,
                         polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto"
                         ) -> Optional[TICKS_DTYPE]:

        """Get ticks for the specified date range from the MetaTrader 5 terminal.

        [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksrange_py)

        Args:
            symbol(str): Financial instrument name, for example, "EURUSD". Required unnamed parameter.
            date_from(datetime): Date of opening of the first bar from the requested sample. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. Required unnamed parameter.
            date_to(datetime): Date, up to which the ticks are requested. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. Required unnamed parameter.
            flags(int): A flag to define the type of the requested ticks. COPY_TICKS_INFO – ticks with Bid and/or Ask changes, COPY_TICKS_TRADE – ticks with changes in Last and Volume, COPY_TICKS_ALL – all ticks. Flag values are described in the COPY_TICKS enumeration. Required unnamed parameter.
            parent_mt5_source (bool): Whether to obtain ticks directly from the `parent_mt5` (directly from the terminal) when set to True. Or, from custom broker's path created by the StrategyTester object.

            polars_collect_engine (str): Engine used by Polars when collecting ticks from custom broker's path. Supported values are:
                - ``"auto"`` (default): Use Polars’ standard in-memory engine and
                    respect the ``POLARS_ENGINE_AFFINITY`` environment variable if set.
                - ``"in-memory"``: Explicitly use the default in-memory engine,
                    optimized with multi-threading and SIMD over Arrow data.
                - ``"streaming"``: Process queries in batches, enabling
                    larger-than-RAM datasets.
                - ``"gpu"``: Use NVIDIA GPUs via RAPIDS cuDF for accelerated execution.
                    Requires installing Polars with GPU support, e.g.:
                    ``pip install polars[gpu] --extra-index-url=https://pypi.nvidia.com``.
        Returns:
            Returns ticks as the numpy array with the named time, bid, ask, last and flags columns. The 'flags' value can be a combination of flags from the TICK_FLAG enumeration. Return None in case of an error. The info on the error can be obtained using last_error().

        Notes:
            - In some cases, copying data directly from the terminal becomes cheap compared to reading from parquet files that introduce `file IO` operations that are computationally expensive. In such cases, `parent_mt5` becomes handy.
        """

        if not isinstance(date_from, datetime) or isinstance(date_to, datetime):
            self.warning_log("Failed, both `date_from` and `date_to` must be datetime objects")
            return None

        if parent_mt5_source:

            ticks = self.parent_mt5.copy_ticks_range(symbol, date_from, date_to, flags)
            self._last_error = self.parent_mt5.last_error()
            return ticks

        return self.history_manager.copy_ticks_range_parquet(symbol=symbol,
                                                             date_from=date_from,
                                                             date_to=date_to,
                                                             polars_collect_engine=polars_collect_engine,
                                                             broker_data_dir=self.broker_data_path,
                                                             flags=flags,
                                                             logger=self.logger)

    @requires_initialization
    def copy_ticks_from(self,
                        symbol: str,
                        date_from: datetime,
                        count: int,
                        flags: int = MetaTrader5.COPY_TICKS_ALL,
                        parent_mt5_source: bool=False,
                        polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto"
                        ) -> Optional[np.array]:

        """Get ticks from the MetaTrader 5 terminal starting from the specified date.

        [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksfrom_py)

        Args:
            symbol(str): Financial instrument name, for example, "EURUSD". Required unnamed parameter.
            date_from(datetime): Date of opening of the first bar from the requested sample. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. Required unnamed parameter.
            count(int): Number of ticks to receive. Required unnamed parameter.
            flags(int): A flag to define the type of the requested ticks. COPY_TICKS_INFO – ticks with Bid and/or Ask changes, COPY_TICKS_TRADE – ticks with changes in Last and Volume, COPY_TICKS_ALL – all ticks. Flag values are described in the COPY_TICKS enumeration. Required unnamed parameter.
            parent_mt5_source (bool): Whether to obtain ticks directly from the `parent_mt5` (directly from the terminal) when set to True. Or, from custom broker's path created by the StrategyTester object.

            polars_collect_engine (str): Engine used by Polars when collecting ticks from custom broker's path. Supported values are:
                - ``"auto"`` (default): Use Polars’ standard in-memory engine and
                    respect the ``POLARS_ENGINE_AFFINITY`` environment variable if set.
                - ``"in-memory"``: Explicitly use the default in-memory engine,
                    optimized with multi-threading and SIMD over Arrow data.
                - ``"streaming"``: Process queries in batches, enabling
                    larger-than-RAM datasets.
                - ``"gpu"``: Use NVIDIA GPUs via RAPIDS cuDF for accelerated execution.
                    Requires installing Polars with GPU support, e.g.:
                    ``pip install polars[gpu] --extra-index-url=https://pypi.nvidia.com``.

        Returns:
            Returns ticks as the numpy array with the named time, bid, ask, last and flags columns. The 'flags' value can be a combination of flags from the TICK_FLAG enumeration. Return None in case of an error. The info on the error can be obtained using last_error().

        Notes:
            - In some cases, copying data directly from the terminal becomes cheap compared to reading from parquet files that introduce `file IO` operations that are computationally expensive. In such cases, `parent_mt5` becomes handy.
        """

        if not isinstance(date_from, datetime):
            self.warning_log("Failed, `date_from` must be a datetime object")
            return None

        if parent_mt5_source:

            ticks = self.parent_mt5.copy_ticks_from(symbol, date_from, count, flags)
            self._last_error = self.parent_mt5.last_error()
            return ticks

        return self.history_manager.copy_ticks_from_parquet(symbol=symbol,
                                                          date_from=date_from,
                                                          limit=count,
                                                          polars_collect_engine=polars_collect_engine,
                                                          broker_data_dir=self.broker_data_path,
                                                          flags=flags,
                                                          logger=self.logger)

    @requires_initialization
    def orders_total(self) -> int:

        """Get the number of active orders.

        Returns (int): The number of active orders in either a simulator or MetaTrader 5, or
                        returns a negative number if there was an error getting the value
        """

        return len(self.ORDERS)

    @requires_initialization
    def orders_get(self,
                   symbol: Optional[str] = None,
                   group: Optional[str] = None,
                   ticket: Optional[int] = None) -> Optional[tuple[TradeOrder]]:

        """Get active orders with the ability to filter by symbol or ticket. There are three call options.

        [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5ordersget_py)

        Args:
            symbol (str | optional): Symbol name. If a symbol is specified, the ticket parameter is ignored.
            group (str | optional): The filter for arranging a group of necessary symbols. If the group is specified, the function returns only active orders meeting a specified criteria for a symbol name.

            ticket (int | optional): Order ticket (ORDER_TICKET).

        Returns:
            list: Returns info in the form of a tuple structure (TradeOrder). Return None in case of an error. The info on the error can be obtained using last_error().
        """

        orders = self.ORDERS

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

    @requires_initialization
    def positions_total(self) -> int:
        """Get the number of open positions in MetaTrader 5 client.

        [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5positionstotal_py)
        Returns:
            int: number of positions
        """
        return len(self.POSITIONS)

    @requires_initialization
    def positions_get(self,
                      symbol: Optional[str] = None,
                      group: Optional[str] = None,
                      ticket: Optional[int] = None) -> tuple[TradePosition]:

        """Get open positions with the ability to filter by symbol or ticket. There are three call options.

        [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5positionsget_py)

        Args:
            symbol (str | optional): Symbol name. If a symbol is specified, the ticket parameter is ignored.
            group (str | optional): The filter for arranging a group of necessary symbols. Optional named parameter. If the group is specified, the function returns only positions meeting a specified criteria for a symbol name.

            ticket (int | optional): Position ticket -> https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer

        Returns:

            list: Returns info in the form of a tuple structure (TradePosition). Return None in case of an error. The info on the error can be obtained using last_error().
        """

        positions = self.POSITIONS

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

    @requires_initialization
    def history_orders_total(self, date_from: datetime, date_to: datetime) -> int:
        """
        Get the number of orders in trading history within the specified interval.

        [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5historyorderstotal_py)

        Args:
            date_from (datetime):
                Start date of the requested history interval.

            date_to (datetime):
                End date of the requested history interval.

        Note:
            `date_from` must be earlier than `date_to`.
        """

        if not isinstance(date_from, datetime) or not isinstance(date_to, datetime):
            raise ValueError("date_from and date_to must be specified")

        date_from_ts = int(date_from.timestamp())
        date_to_ts = int(date_to.timestamp())

        return sum(
            1
            for o in self.ORDERS_HISTORY
            if date_from_ts <= o.time_setup <= date_to_ts
        )

    @requires_initialization
    def history_orders_get(self,
                           date_from: Optional[datetime] = None,
                           date_to: Optional[datetime] = None,
                           group: Optional[str] = None,
                           ticket: Optional[int] = None,
                           position: Optional[int] = None
                           ) -> Optional[tuple[TradeOrder]]:
        """
          Get orders from trading history, with optional filtering by symbol group,
          order ticket, or position ticket.

          [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5historyordersget_py)

          Args:
              date_from (datetime | None, optional):
                  Start of the requested history interval.

              date_to (datetime | None, optional):
                  End of the requested history interval.

              group (str | None, optional):
                  Symbol filter applied to the date-range query. MT5 supports masks
                  with `*`, multiple comma-separated conditions, and exclusion with
                  `!`. Inclusion conditions should come before exclusions. Example:
                  `"*, !EUR"`. :contentReference[oaicite:1]{index=1}

              ticket (int | None, optional):
                  Order ticket to retrieve. When provided, this method returns
                  orders matching that ticket.

              position (int | None, optional):
                  Position ticket used to retrieve all orders whose
                  `ORDER_POSITION_ID` matches that position. :contentReference[oaicite:2]{index=2}

          Returns:
              tuple[TradeOrder] | None:
                  A tuple of `TradeOrder` records. Returns `None` on error.

          Raises:
              ValueError:
                  If `date_from` or `date_to` is not a `datetime` instance.
        """

        if not isinstance(date_from, datetime) or not isinstance(date_to, datetime):
            raise ValueError("date_from and date_to must be specified")

        orders = self.ORDERS_HISTORY

        # ticket filter (highest priority)
        if ticket is not None:
            return tuple(o for o in orders if o.ticket == ticket)

        # position filter
        if position is not None:
            return tuple(o for o in orders if o.position_id == position)

        # date range is a requirement
        if date_from is None or date_to is None:
            self.error_log("date_from and date_to must be specified")
            return None

        date_from_ts = int(date_from.timestamp())
        date_to_ts = int(date_to.timestamp())

        filtered = (
            o for o in orders
            if date_from_ts <= o.time_setup <= date_to_ts
        )  # obtain orders that fall within this time range

        # optional group filter
        if group is not None:
            filtered = (
                o for o in filtered
                if fnmatch.fnmatch(o.symbol, group)
            )

        return tuple(filtered)

    @requires_initialization
    def history_deals_total(self, date_from: datetime, date_to: datetime) -> int:
        """
        Get the number of deals in history within the specified date range.

        [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5historydealstotal_py)

        Args:
            date_from (datetime):
                Date the orders are requested from. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01.

            date_to (datetime, required):
                Date, up to which the orders are requested. Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01.

        Returns:
            An integer value.
        """

        if isinstance(date_from, (int, float)):
            date_from = datetime.fromtimestamp(date_from)
        if isinstance(date_to, (int, float)):
            date_to = datetime.fromtimestamp(date_to)

        date_from_ts = int(date_from.timestamp())
        date_to_ts = int(date_to.timestamp())

        return sum(
            1
            for d in self.DEALS
            if date_from_ts <= d.time <= date_to_ts
        )

    @requires_initialization
    def history_deals_get(self,
                          date_from: datetime,
                          date_to: datetime,
                          group: Optional[str] = None,
                          ticket: Optional[int] = None,
                          position: Optional[int] = None
                          ) -> Optional[tuple[TradeDeal]]:
        """Gets deals from trading history within the specified interval with the ability to filter by ticket or position.

        [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5historydealsget_py)

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

        if isinstance(date_from, (int, float)):
            date_from = datetime.fromtimestamp(date_from)
        if isinstance(date_to, (int, float)):
            date_to = datetime.fromtimestamp(date_to)

        deals = self.DEALS

        # ticket filter (highest priority)
        if ticket is not None:
            return tuple(d for d in deals if d.ticket == ticket)

        # position filter
        if position is not None:
            return tuple(d for d in deals if d.position_id == position)

        # date range is a requirement
        if date_from is None or date_to is None:
            self.error_log("date_from and date_to must be specified")
            return None

        date_from_ts = int(date_from.timestamp())
        date_to_ts = int(date_to.timestamp())

        filtered = (
            d for d in deals
            if date_from_ts <= d.time <= date_to_ts
        )  # obtain orders that fall within this time range

        # optional group filter
        if group is not None:
            filtered = (
                d for d in filtered
                if fnmatch.fnmatch(d.symbol, group)
            )

        return tuple(filtered)

    @staticmethod
    def _calc_commission() -> float:
        """
        MT5-style commission calculation.
        """

        return -0.2

    @staticmethod
    def _position_to_order(position: TradePosition, ticket=None) -> TradeOrder:
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
            type_time=VirtualMetaTrader5.ORDER_TIME_GTC,
            type_filling=VirtualMetaTrader5.ORDER_FILLING_FOK,
            state=VirtualMetaTrader5.ORDER_STATE_FILLED,

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

    @staticmethod
    def _build_trade_request(request: dict) -> TradeRequest:
        return TradeRequest(
            action=request.get("action", 0),
            magic=request.get("magic", 0),
            order=request.get("order", 0),
            symbol=request.get("symbol", ""),
            volume=float(request.get("volume", 0.0)),
            price=float(request.get("price", 0.0)),
            stoplimit=float(request.get("stoplimit", 0.0)),
            sl=float(request.get("sl", 0.0)),
            tp=float(request.get("tp", 0.0)),
            deviation=int(request.get("deviation", 0)),
            type=request.get("type", 0),
            type_filling=request.get("type_filling", 0),
            type_time=request.get("type_time", 0),
            expiration=int(request.get("expiration", 0)),
            comment=str(request.get("comment", "")),
            position=int(request.get("position", 0)),
            position_by=int(request.get("position_by", 0)),
        )

    @requires_initialization
    def _make_result(
            self,
            request: TradeRequest,
            retcode: int,
            deal: int = 0,
            order: int = 0,
            volume: float = 0.0,
    ) -> OrderSendResult:

        ticks = self.symbol_info_tick(request.symbol)

        return OrderSendResult(
            retcode=retcode,
            deal=deal,
            order=order,
            volume=volume,
            price=request.price,
            bid=ticks.bid if ticks else 0.0,
            ask=ticks.ask if ticks else 0.0,
            comment=self.RETCODE_MAP.get(retcode, "Unknown"),
            request_id=0,
            retcode_external=0,
            request=request,
        )

    @requires_initialization
    def _get_trade_validators(self, symbol: str):
        """Returns TradeValidators instance for the given symbol."""

        if symbol not in self.TRADE_VALIDATORS_CACHE:
            self.TRADE_VALIDATORS_CACHE[symbol] = TradeValidators(
                symbol_info=self.symbol_info(symbol),
                logger=self.logger
            )

        return self.TRADE_VALIDATORS_CACHE[symbol]

    @requires_initialization
    def _create_position_from_request(self, time: int, time_msc: float, request: TradeRequest,
                                      margin: Optional[float] = 0.0) -> TradePosition:

        ticket = self._generate_position_ticket()

        return TradePosition(
            ticket=ticket,
            time=time,
            time_msc=time_msc,
            time_update=time,
            time_update_msc=time_msc,
            type=request.type,
            magic=request.magic,
            identifier=0,
            reason="",
            volume=request.volume,
            price_open=request.price,
            sl=request.sl,
            tp=request.tp,
            price_current=request.price,
            swap=0,
            profit=0,
            symbol=request.symbol,
            comment=request.comment,
            external_id=0,

            # ---- additional fields ----
            # last_swap_time,
            margin=margin,
        )

    @requires_initialization
    def _create_order_from_request(self, time: int, time_msc: int, request: TradeRequest,
                                   state: int = MetaTrader5Constants.ORDER_STATE_PLACED) -> TradeOrder:

        ticket = self._generate_order_ticket()

        return TradeOrder(
            ticket=ticket,
            time_setup=time,
            time_setup_msc=time_msc,
            time_done=0,
            time_done_msc=0,
            time_expiration=0,  # GTC
            type=request.type,
            type_time=getattr(request, "type_time", 0),  # ORDER_TIME_GTC
            type_filling=getattr(request, "type_filling", 0),  # FOK/IOC/RETURN
            state=state,
            magic=request.magic,
            position_id=0,
            position_by_id=0,
            reason="",
            volume_initial=request.volume,
            volume_current=request.volume,
            price_open=request.price,
            sl=request.sl,
            tp=request.tp,
            price_current=request.price,
            price_stoplimit=0.0,  # only for STOP_LIMIT (future support)
            symbol=request.symbol,
            comment=request.comment,
            external_id=0,
        )

    @requires_initialization
    def _create_deal_from_request(self, time: int, time_msc: float, entry: int, request: TradeRequest,
                                  position: TradePosition, commission: float = 0.0, swap: float = 0.0,
                                  fee: float = 0.0) -> TradeDeal:

        ticket = self._generate_deal_ticket()
        order = self._generate_order_ticket()

        digits = self.symbol_info(position.symbol).digits

        reason: int = self.DEAL_REASON_EXPERT
        price = request.price
        sl = position.sl
        tp = position.tp

        if round(price, digits) != round(sl, digits):  # take profit is hit
            reason = self.DEAL_REASON_SL

        if round(price, digits) != round(tp, digits):  # stop loss is hit
            reason = self.DEAL_REASON_TP

        return TradeDeal(
            ticket=ticket,
            order=order,
            time=time,
            time_msc=time_msc,
            type=request.type,
            entry=entry,
            magic=request.magic,
            position_id=position.ticket,
            reason=reason,
            volume=request.volume,
            price=request.price,
            commission=commission,
            swap=swap,
            profit=position.profit,
            fee=fee,
            symbol=request.symbol,
            comment=request.comment,
            external_id=0,

            balance=self.ACCOUNT.balance
        )

    @requires_initialization
    def _open_position(self, request: dict):

        trade_request = self._build_trade_request(request=request)

        try:
            # Necessary parameters for opening a position
            order_type = request.get("type")
            symbol = request.get("symbol")
            volume = float(request.get("volume"))
            price = float(request.get("price"))

            sl = float(request.get("sl", 0))
            tp = float(request.get("tp", 0))

        except KeyError:
            self.debug_log(f"{return_code_description(self.TRADE_RETCODE_INVALID)}")
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID)

        if order_type not in (self.ORDER_TYPE_BUY, self.ORDER_TYPE_SELL):
            self.debug_log(f"{return_code_description(self.TRADE_RETCODE_INVALID)}")
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID)

        # ------------------------ All checks and return codes ----------------------

        # checking if the price is valid

        tick = self.symbol_info_tick(symbol)
        symbol_info = self.symbol_info(symbol)
        validators = self._get_trade_validators(symbol=symbol)
        ac_info = self.account_info()

        if tick is None:
            self.debug_log(f"{return_code_description(self.TRADE_RETCODE_PRICE_OFF)}")
            return self._make_result(trade_request, self.TRADE_RETCODE_PRICE_OFF)  # NO_QUOTES

        eps = pow(10, -symbol_info.digits)

        if type == self.ORDER_TYPE_BUY:
            if not TradeValidators.price_equal(price, tick.ask, eps):
                self.debug_log(f"{return_code_description(self.TRADE_RETCODE_PRICE_CHANGED)}")
                return self._make_result(trade_request, self.TRADE_RETCODE_PRICE_CHANGED)  # PRICE_CHANGED

        elif type == self.ORDER_TYPE_SELL:
            if not TradeValidators.price_equal(price, tick.bid, eps):
                self.debug_log(f"{return_code_description(self.TRADE_RETCODE_PRICE_CHANGED)}")
                return self._make_result(trade_request, self.TRADE_RETCODE_PRICE_CHANGED)

        # sl and tp checks
        if not validators.is_valid_sl(price, sl, order_type):
            self.debug_log(f"{return_code_description(self.TRADE_RETCODE_INVALID_STOPS)}")
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID_STOPS)

        if not validators.is_valid_tp(price, tp, order_type):
            self.debug_log(f"{return_code_description(self.TRADE_RETCODE_INVALID_STOPS)}")
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID_STOPS)

        # check if there is enough money

        price_for_margin = tick.ask if order_type == self.ORDER_TYPE_BUY else tick.bid

        margin = self.order_calc_margin(
            order_type=order_type,
            symbol=symbol,
            volume=volume,
            price=price_for_margin
        )

        future_margin = ac_info.margin + margin
        future_equity = ac_info.equity

        if future_margin > 0:
            future_margin_level = (future_equity / future_margin) * 100
        else:
            future_margin_level = float("inf")

        if future_margin_level <= ac_info.margin_so_call:
            self.debug_log(f"{return_code_description(self.TRADE_RETCODE_NO_MONEY)}")
            return self._make_result(trade_request, self.TRADE_RETCODE_NO_MONEY)

        if margin > ac_info.margin_free:
            self.debug_log(f"{return_code_description(self.TRADE_RETCODE_NO_MONEY)}")
            return self._make_result(trade_request, self.TRADE_RETCODE_NO_MONEY)  # NO_MONEY

        # ---------------- MAX ORDERS CHECK ---------------------

        if validators.is_max_orders_reached(open_orders=self.orders_total(), ac_limit_orders=self.ACCOUNT.limit_orders):
            self.debug_log(f"{return_code_description(self.TRADE_RETCODE_LIMIT_ORDERS)}")
            return self._make_result(trade_request, self.TRADE_RETCODE_LIMIT_ORDERS)

        # ---------------- VOLUME VALIDATION ----------------

        if not validators.is_valid_lotsize(volume):
            self.debug_log(f"{return_code_description(self.TRADE_RETCODE_INVALID_VOLUME)}")
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID_VOLUME)

        total_volume = sum([pos.volume for pos in self.POSITIONS]) + sum(
            [order.volume_current for order in self.ORDERS])

        if validators.is_symbol_volume_reached(symbol_volume=total_volume, volume_limit=symbol_info.volume_limit):
            self.debug_log(f"{return_code_description(self.TRADE_RETCODE_LIMIT_VOLUME)}")
            return self._make_result(trade_request, self.TRADE_RETCODE_LIMIT_VOLUME)

        # ------------------------ FILL THE REQUEST ----------------------------------

        time = tick.time
        time_msc = tick.time_msc

        # craft a position and add it to history container/array
        position = self._create_position_from_request(time=time, time_msc=time_msc, request=trade_request,
                                                      margin=margin)
        self.POSITIONS.append(position)

        # craft a deal and add it to history
        deal = self._create_deal_from_request(time=time, time_msc=time_msc, entry=self.DEAL_ENTRY_IN,
                                              request=trade_request, position=position)
        self.DEALS.append(deal)

        # store history of orders
        self.ORDERS_HISTORY.append(
            self._position_to_order(position=position, ticket=self._generate_order_history_ticket())
        )

        self.info_log(f"Position {deal.ticket} opened successfully!")
        return self._make_result(trade_request, retcode=self.TRADE_RETCODE_DONE, deal=deal.ticket, order=deal.order,
                                 volume=position.volume)

    @requires_initialization
    def _close_position(self, request: dict):

        trade_request = self._build_trade_request(request=request)

        try:
            symbol = request.get("symbol")
            volume = float(request.get("volume"))
            position_id = request.get("position")
        except KeyError:
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID)

        tick = self.symbol_info_tick(symbol)
        if tick is None:
            return self._make_result(trade_request, self.TRADE_RETCODE_PRICE_OFF)

        # ---------------- FIND POSITION ----------------

        position = None

        if position_id is not None:
            for pos in self.POSITIONS:
                if pos.ticket == position_id:
                    position = pos
                    break

        """
        else:
            # fallback: find by symbol (netting behavior)
            for pos in self.POSITIONS:
                if pos.symbol == symbol:
                    position = pos
                    break
        """

        if position is None:
            return self._make_result(trade_request, self.TRADE_RETCODE_POSITION_CLOSED)

        # ---------------- VALIDATE VOLUME ----------------

        if volume <= 0 or volume > position.volume:
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID_VOLUME)

        # ---------------- DETERMINE CLOSE PRICE ----------------

        if position.type == self.ORDER_TYPE_BUY:
            close_price = tick.bid
        else:
            close_price = tick.ask

        # ---------------- CALCULATE PROFIT ----------------

        profit = self.order_calc_profit(
            position.type,
            symbol,
            volume,
            position.price_open,
            close_price,
        )

        # ---------------- UPDATE ACCOUNT ----------------

        acct = self.ACCOUNT

        new_balance = acct.balance + profit
        new_equity = acct.equity + profit
        released_margin = position.margin * (volume / position.volume)

        new_margin = acct.margin - released_margin
        new_margin_free = new_equity - new_margin

        self.ACCOUNT = acct._replace(
            balance=new_balance,
            equity=new_equity,
            profit=acct.profit + profit,
            margin=new_margin,
            margin_free=new_margin_free,
            margin_level=(new_equity / new_margin * 100) if new_margin > 0 else float("inf"),
        )

        # ---------------- UPDATE POSITION ----------------

        idx = next(
            (i for i, o in enumerate(self.ORDERS_HISTORY)
             if o.type == position.type and o.position_id == position.ticket),
            None
        )

        if idx is not None:
            self.ORDERS_HISTORY[idx] = self.ORDERS_HISTORY[idx]._replace(
                time_done=self.current_time(),
                time_done_msc=int(self.current_time() * 1000),
                volume_current=position.volume,
                price_current=position.price_current,
            )

        if volume == position.volume:
            # FULL CLOSE
            self.POSITIONS.remove(position)
        else:
            # PARTIAL CLOSE
            remaining_volume = position.volume - volume
            position = position._replace(volume=remaining_volume)
            # replace in list
            for i, pos in enumerate(self.POSITIONS):
                if pos.ticket == position.ticket:
                    self.POSITIONS[i] = position
                    break

        # ---------------- CREATE DEAL ----------------

        time = tick.time
        time_msc = tick.time_msc

        deal = self._create_deal_from_request(
            time=time,
            time_msc=time_msc,
            entry=self.DEAL_ENTRY_OUT,
            request=trade_request,
            position=position,
            commission=self._calc_commission()
        )

        self.DEALS.append(deal)

        self.info_log(f"Position {deal.ticket} closed successfully!")
        return self._make_result(
            trade_request,
            retcode=self.TRADE_RETCODE_DONE,
            deal=deal.ticket,
            order=deal.order,
            volume=volume,
        )

    @requires_initialization
    def _modify_position(self, request: dict):

        trade_request = self._build_trade_request(request=request)

        try:
            symbol = request.get("symbol")
            position_id = request.get("position")
            sl = float(request.get("sl", 0))
            tp = float(request.get("tp", 0))
        except KeyError:
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID)

        tick = self.symbol_info_tick(symbol)
        if tick is None:
            return self._make_result(trade_request, self.TRADE_RETCODE_PRICE_OFF)

        symbol_info = self.symbol_info(symbol)
        digits = symbol_info.digits
        validators = self._get_trade_validators(symbol)

        # ---------------- FIND POSITION ----------------

        position = None

        if position_id is not None:
            for pos in self.POSITIONS:
                if pos.ticket == position_id:
                    position = pos
                    break

        if position is None:
            return self._make_result(trade_request, self.TRADE_RETCODE_POSITION_CLOSED)

        # prevent useless modification
        if (
                round(sl, digits) == round(position.sl, digits) and
                round(tp, digits) == round(position.tp, digits)
        ):
            return self._make_result(trade_request, self.TRADE_RETCODE_DONE)

        # ---------------- VALIDATE SL / TP ----------------

        # Use current market price as reference
        price_ref = tick.ask if position.type == self.ORDER_TYPE_BUY else tick.bid

        if sl != 0:
            if not validators.is_valid_sl(price_ref, sl, position.type):
                return self._make_result(trade_request, self.TRADE_RETCODE_INVALID_STOPS)

        if tp != 0:
            if not validators.is_valid_tp(price_ref, tp, position.type):
                return self._make_result(trade_request, self.TRADE_RETCODE_INVALID_STOPS)

        # ---------------- APPLY MODIFICATION ----------------

        updated_position = position._replace(
            sl=sl if sl != 0 else position.sl,
            tp=tp if tp != 0 else position.tp,
        )

        # replace in list
        for i, pos in enumerate(self.POSITIONS):
            if pos.ticket == position.ticket:
                self.POSITIONS[i] = updated_position
                break

        self.info_log(f"Position {position.ticket} modified successfully!")

        return self._make_result(
            trade_request,
            retcode=self.TRADE_RETCODE_DONE,
            order=position.ticket,
        )

    @requires_initialization
    def _open_pending_order(self, request: dict):

        trade_request = self._build_trade_request(request=request)

        try:
            order_type = request.get("type")
            symbol = request.get("symbol")
            volume = float(request.get("volume"))
            price = float(request.get("price"))

            sl = float(request.get("sl", 0))
            tp = float(request.get("tp", 0))

        except KeyError:
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID)

        # ---------------- TYPE CHECK ----------------

        if order_type not in (
                self.ORDER_TYPE_BUY_LIMIT,
                self.ORDER_TYPE_SELL_LIMIT,
                self.ORDER_TYPE_BUY_STOP,
                self.ORDER_TYPE_SELL_STOP,
        ):
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID)

        tick = self.symbol_info_tick(symbol)
        symbol_info = self.symbol_info(symbol)
        validators = self._get_trade_validators(symbol)

        if tick is None:
            return self._make_result(trade_request, self.TRADE_RETCODE_PRICE_OFF)

        # ---------------- PRICE VALIDATION ----------------

        if not validators.is_valid_pending_price(price, tick, order_type):
            self.debug_log(
                f"Invalid price for: {MetaTrader5Constants.ORDER_TYPE_MAP[order_type]} price: {price} ask: {tick.ask}")
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID_PRICE)

        # ---------------- SL / TP VALIDATION ----------------

        if sl != 0:
            if not validators.is_valid_sl(price, sl, order_type):
                return self._make_result(trade_request, self.TRADE_RETCODE_INVALID_STOPS)

        if tp != 0:
            if not validators.is_valid_tp(price, tp, order_type):
                return self._make_result(trade_request, self.TRADE_RETCODE_INVALID_STOPS)

        # ---------------- MAX ORDERS CHECK ---------------------

        if validators.is_max_orders_reached(open_orders=self.orders_total(), ac_limit_orders=self.ACCOUNT.limit_orders):
            return self._make_result(trade_request, self.TRADE_RETCODE_LIMIT_ORDERS)

        # ---------------- VOLUME VALIDATION ----------------

        if not validators.is_valid_lotsize(volume):
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID_VOLUME)

        total_volume = sum([pos.volume for pos in self.POSITIONS]) + sum(
            [order.volume_current for order in self.ORDERS])

        if validators.is_symbol_volume_reached(symbol_volume=total_volume, volume_limit=symbol_info.volume_limit):
            return self._make_result(trade_request, self.TRADE_RETCODE_LIMIT_VOLUME)

        # ---------------- CREATE ORDER ----------------

        tick_time = tick.time
        tick_time_msc = tick.time_msc

        order = self._create_order_from_request(
            time=tick_time,
            time_msc=tick_time_msc,
            request=trade_request,
        )

        self.ORDERS.append(order)

        # store history of orders
        self.ORDERS_HISTORY.append(order)
        self.info_log(f"Pending order {order.ticket} created successfully!")

        return self._make_result(
            trade_request,
            retcode=self.TRADE_RETCODE_DONE,
            order=order.ticket,
            volume=volume,
        )

    @requires_initialization
    def _modify_order(self, request: dict):

        trade_request = self._build_trade_request(request=request)

        try:
            order_id = request.get("order")
            symbol = request.get("symbol")

            new_price = float(request.get("price", 0))
            sl = float(request.get("sl", 0))
            tp = float(request.get("tp", 0))

        except KeyError:
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID)

        tick = self.symbol_info_tick(symbol)
        if tick is None:
            return self._make_result(trade_request, self.TRADE_RETCODE_PRICE_OFF)

        symbol_info = self.symbol_info(symbol)
        digits = symbol_info.digits
        validators = self._get_trade_validators(symbol)

        # ---------------- FIND ORDER ----------------

        order = None
        for o in self.ORDERS:
            if o.ticket == order_id:
                order = o
                break

        if order is None:
            self.debug_log(f"Invalid order ticket = {order_id}")
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID_ORDER)

        # prevent useless modification
        if (
                round(new_price, digits) == round(order.price_open, digits) and
                round(sl, digits) == round(order.sl, digits) and
                round(tp, digits) == round(order.tp, digits)
        ):
            return self._make_result(trade_request, self.TRADE_RETCODE_DONE)

        # ---------------- CHECK TYPE ----------------

        if order.type not in (
                self.ORDER_TYPE_BUY_LIMIT,
                self.ORDER_TYPE_SELL_LIMIT,
                self.ORDER_TYPE_BUY_STOP,
                self.ORDER_TYPE_SELL_STOP,
        ):
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID)

        # ---------------- VALIDATE PRICE ----------------

        price = new_price if new_price != 0 else order.price_open

        if not validators.is_valid_pending_price(price, tick, order.type):
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID_PRICE)

        # ---------------- VALIDATE SL / TP ----------------

        if sl != 0:
            if not validators.is_valid_sl(price, sl, order.type):
                return self._make_result(trade_request, self.TRADE_RETCODE_INVALID_STOPS)

        if tp != 0:
            if not validators.is_valid_tp(price, tp, order.type):
                return self._make_result(trade_request, self.TRADE_RETCODE_INVALID_STOPS)

        # ---------------- APPLY MODIFICATION ----------------

        updated_order = order._replace(
            price_open=price,
            sl=sl if sl != 0 else order.sl,
            tp=tp if tp != 0 else order.tp,
        )

        # replace in list
        for i, o in enumerate(self.ORDERS):
            if o.ticket == order.ticket:
                self.ORDERS[i] = updated_order
                break

        # ---------------- VISUAL HISTORY ----------------

        time = tick.time

        self.info_log(f"Pending order {order.ticket} modified successfully!")

        return self._make_result(
            trade_request,
            retcode=self.TRADE_RETCODE_DONE,
            order=order.ticket,
        )

    @requires_initialization
    def _delete_order(self, request: dict):

        trade_request = self._build_trade_request(request=request)

        try:
            order_id = request.get("order")
            symbol = request.get("symbol")
        except KeyError:
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID)

        tick = self.symbol_info_tick(symbol)
        if tick is None:
            return self._make_result(trade_request, self.TRADE_RETCODE_PRICE_OFF)

        # ---------------- FIND ORDER ----------------

        order = None
        for o in self.ORDERS:
            if o.ticket == order_id:
                order = o
                break

        if order is None:
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID_ORDER)

        # ---------------- VALIDATE STATE ----------------

        if order.state != self.ORDER_STATE_PLACED:
            return self._make_result(trade_request, self.TRADE_RETCODE_INVALID)

        # ---------------- UPDATE ORDER STATE ----------------

        time = tick.time
        time_msc = tick.time_msc

        canceled_order = order._replace(
            state=self.ORDER_STATE_CANCELED,
            time_done=time,
            time_done_msc=time_msc,
        )

        # ---------------- REMOVE FROM ACTIVE ORDERS ----------------

        self.ORDERS = [o for o in self.ORDERS if o.ticket != order.ticket]

        # ---------------- OPTIONAL: STORE HISTORY ----------------

        self.ORDERS_HISTORY.append(canceled_order)
        self.info_log(f"Pending order {order.ticket} deleted successfully!")

        return self._make_result(
            trade_request,
            retcode=self.TRADE_RETCODE_DONE,
            order=order.ticket,
        )

    @requires_initialization
    def order_send(self, request: dict) -> Optional[OrderSendResult]:
        action = request.get("action")

        if action == self.TRADE_ACTION_DEAL:
            if request.get("position"):  # their subtle difference is a position
                return self._close_position(request)
            else:
                return self._open_position(request)

        elif action == self.TRADE_ACTION_SLTP:
            return self._modify_position(request)

        elif action == self.TRADE_ACTION_PENDING:
            return self._open_pending_order(request)

        elif action == self.TRADE_ACTION_MODIFY:
            return self._modify_order(request)

        elif action == self.TRADE_ACTION_REMOVE:
            return self._delete_order(request)

        self.critical_log("Unknown trade action")
        return None

    @requires_initialization
    def _terminate_all_positions(self, comment: str) -> bool:

        for pos in self.positions_get():

            position_type = pos.type  # 0=BUY, 1=SELL

            # Get close price (BID for buy, ASK for sell)

            tick_info = self.symbol_info_tick(pos.symbol)
            price = tick_info.bid if position_type == self.POSITION_TYPE_BUY else tick_info.ask

            # Set close order type
            order_type = self.ORDER_TYPE_SELL if position_type == self.POSITION_TYPE_BUY else self.ORDER_TYPE_BUY

            request = {
                "action": self.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "magic": pos.magic,
                "type": order_type,
                "price": price,
                "deviation": 1000,
                "type_time": self.ORDER_TIME_GTC,
                "comment": comment
            }

            # Send the close request

            if self.order_send(request) is None:
                return False

        return True

    @requires_initialization
    def order_calc_profit(self,
                          order_type: int,
                          symbol: str,
                          volume: float,
                          price_open: float,
                          price_close: float) -> float:
        """
        Return profit in the account currency for a specified trading operation.

        [Reference](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercalcprofit_py)

        Args:
            order_type (int): The type of position taken, either 0 (buy) or 1 (sell).
            symbol (str): Financial instrument name.
            volume (float):   Trading operation volume.
            price_open (float): Open Price.
            price_close (float): Close Price.
        """

        sym = self.symbol_info(symbol)

        contract_size = sym.trade_contract_size

        direction = 0

        # --- Determine direction ---
        if order_type in self.BUY_ACTIONS:
            direction = 1
        elif order_type in self.SELL_ACTIONS:
            direction = -1

        # --- Core profit calculation ---

        calc_mode = sym.trade_calc_mode
        price_delta = (price_close - price_open) * direction

        try:
            # ------------------ FOREX / CFD / STOCKS -----------------------
            if calc_mode in (
                    self.SYMBOL_CALC_MODE_FOREX,
                    self.SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE,
                    self.SYMBOL_CALC_MODE_CFD,
                    self.SYMBOL_CALC_MODE_CFDINDEX,
                    self.SYMBOL_CALC_MODE_CFDLEVERAGE,
                    self.SYMBOL_CALC_MODE_EXCH_STOCKS,
                    self.SYMBOL_CALC_MODE_EXCH_STOCKS_MOEX,
            ):
                profit = price_delta * contract_size * volume

            # ---------------- FUTURES --------------------
            elif calc_mode in (
                    self.SYMBOL_CALC_MODE_FUTURES,
                    self.SYMBOL_CALC_MODE_EXCH_FUTURES,
                    # SYMBOL_CALC_MODE_EXCH_FUTURES_FORTS,
            ):
                tick_value = sym.trade_tick_value
                tick_size = sym.trade_tick_size

                if tick_size <= 0:
                    self.critical_log("Invalid tick size")
                    return 0.0

                profit = price_delta * volume * (tick_value / tick_size)

            # ---------- BONDS -------------------

            elif calc_mode in (
                    self.SYMBOL_CALC_MODE_EXCH_BONDS,
                    self.SYMBOL_CALC_MODE_EXCH_BONDS_MOEX,
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
            elif calc_mode == self.SYMBOL_CALC_MODE_SERV_COLLATERAL:
                liquidity_rate = sym.trade_liquidity_rate
                market_price = (
                    self.TICK_CACHE[symbol].ask if order_type == self.ORDER_TYPE_BUY else
                    self.TICK_CACHE[symbol].bid
                )

                profit = (
                        volume
                        * contract_size
                        * market_price
                        * liquidity_rate
                )

            else:
                self.critical_log(
                    f"Unsupported trade calc mode: {calc_mode}"
                )
                return 0.0

            return round(profit, 2)

        except Exception as e:
            self.critical_log(f"Failed: {e}")
            return 0.0

    @requires_initialization
    def order_calc_margin(self, order_type: int, symbol: str, volume: float, price: float) -> float:
        """
        Return margin in the account currency to perform a specified trading operation.

        """

        if order_type not in (self.ORDER_TYPE_BUY, self.ORDER_TYPE_SELL):
            self.critical_log(f"Invalid order type: {order_type}")
            return 0.0

        if volume <= 0 or price <= 0:
            self.error_log("order_calc_margin failed: invalid volume or price")
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

        if margin_rate <= 0:  # if margin rate is zero set it to 1
            margin_rate = 1.0

        mode = sym.trade_calc_mode

        if mode == self.SYMBOL_CALC_MODE_FOREX:

            base = sym.currency_base
            quote = sym.currency_profit
            account_currency = self.account_info().currency

            # margin = (volume * contract_size * price) / leverage

            if account_currency == base:
                # USDJPY, account USD
                margin = (volume * contract_size) / leverage

            elif account_currency == quote:
                # EURUSD, account USD
                margin = (volume * contract_size * price) / leverage

            else:
                # Cross currency (e.g. EURGBP, account USD)
                # convert margin to account currency
                margin = (volume * contract_size * price) / leverage

                """
                conversion_symbol = f"{quote}{account_currency}"
                conversion_tick = self.symbol_info_tick(conversion_symbol)

                if conversion_tick:
                    margin *= conversion_tick.bid
                else:
                    self.warning_log("Conversion symbol not found")
                """

        elif mode == self.SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE:
            margin = volume * contract_size * price

        elif mode in (
                self.SYMBOL_CALC_MODE_CFD,
                self.SYMBOL_CALC_MODE_CFDINDEX,
                self.SYMBOL_CALC_MODE_EXCH_STOCKS,
                self.SYMBOL_CALC_MODE_EXCH_STOCKS_MOEX,
        ):
            margin = volume * contract_size * price * margin_rate

        elif mode == self.SYMBOL_CALC_MODE_CFDLEVERAGE:
            margin = (volume * contract_size * price * margin_rate) / leverage

        elif mode in (
                self.SYMBOL_CALC_MODE_FUTURES,
                self.SYMBOL_CALC_MODE_EXCH_FUTURES,
                # SYMBOL_CALC_MODE_EXCH_FUTURES_FORTS,
        ):
            margin = volume * sym.margin_initial

        elif mode in (
                self.SYMBOL_CALC_MODE_EXCH_BONDS,
                self.SYMBOL_CALC_MODE_EXCH_BONDS_MOEX,
        ):
            margin = (
                    volume
                    * contract_size
                    * sym.trade_face_value
                    * price
                    / 100
            )

        elif mode == self.SYMBOL_CALC_MODE_SERV_COLLATERAL:
            margin = 0.0

        else:
            self.warning_log(f"Unknown calc mode {mode}, fallback margin formula used")
            margin = (volume * contract_size * price) / leverage

        return round(margin, 2)
