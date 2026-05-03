from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Union


class CSymbolInfo:
    """
        A lightweight Python wrapper that resembles the MQL5 Standard Library class
        `CSymbolInfo` and provides convenient, read-only access to MetaTrader 5
        symbol properties.

        This class stores a symbol name, cached symbol information, and cached tick
        information. Symbol data can be refreshed via `refresh()`, while quote/tick
        data can be refreshed via `refresh_rates()`.

        [Reference (MQL5)](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo)
    """

    def __init__(self, symbol: str, terminal: Any):

        """
        Instantiates the CSymbolInfo class (object).

        Args:
            terminal : MetaTrader5 module-like or the StrategyTester instance.

        Notes:
            - `name(symbol_name)` assigns the symbol and refreshes its static info.
            - `refresh_rates()` updates the cached tick/quote values.
            - If tick data is not refreshed yet, tick-related properties return their default cached values.
            - Time values are returned as timezone-aware UTC datetimes where applicable.

            Method groups mirror the MQL5 layout:<br>

            - Controlling: Refresh, RefreshRates
            - Properties: Name, Select, IsSynchronized
            - Volumes: Volume, VolumeHigh, VolumeLow
            - Miscellaneous: Time, Spread, SpreadFloat, TicksBookDepth
            - Levels: StopsLevel, FreezeLevel
            - Bid/Ask/Last prices
            - Trade modes
            - Swaps
            - Margins and flags
            - Quantization
            - Contract sizes
            - Text properties
            - Session properties
            - Generic accessors: InfoInteger, InfoDouble, InfoString
            - Service functions: NormalizePrice
        """

        self.terminal = terminal
        self._symbol: str = symbol
        self._info: Optional[Any] = None

        self._tick_info: dict[str, Any] = {
            "time": None,
            "ask": 0.0,
            "bid": 0.0,
            "last": 0.0,
            "volume": 0,
            "time_msc": 0,
            "volume_real": 0.0,
        }

        self.check_market_watch()
        self.refresh()

    def refresh(self) -> bool:
        """
        Refreshes the symbol data by fetching symbol information from the terminal.

        Returns:
            True if successful

        Raises:
            RuntimeError: If symbol information could not be retrieved.
        """
        info = self.terminal.symbol_info(self._symbol)
        if not info:
            code, msg = self.terminal.last_error()
            raise RuntimeError(f"Failed to get symbol info for {self._symbol}. MetaTrader5 error = [{code}] {msg}")

        self._info = info
        return True

    def refresh_rates(self) -> bool:
        """
        Refreshes the symbol quotes/gets the latest tick information from the market using `symbol_info_tick(symbol)`.

        Returns:
            True if successful, False otherwise.
        """
        try:
            new_ticks = self.terminal.symbol_info_tick(self._symbol)
            if new_ticks is None:
                return False

            self._tick_info["time"] = new_ticks.time
            self._tick_info["ask"] = new_ticks.ask
            self._tick_info["bid"] = new_ticks.bid
            self._tick_info["last"] = new_ticks.last
            self._tick_info["volume"] = new_ticks.volume
            self._tick_info["time_msc"] = new_ticks.time_msc
            self._tick_info["volume_real"] = new_ticks.volume_real

            return True

        except AttributeError:
            return False

    # ----------- Basic properties ---------------

    @property
    def symbol(self) -> str:
        """Gets the current symbol name."""
        return self._symbol

    def select(self, select: bool = True) -> bool:
        """
        Gets/sets the Market Watch symbol flag.

        Args:
            select : True to select the symbol in Market Watch, False to deselect.

        Returns:
            True if successful, otherwise False.
        """
        return self.terminal.symbol_select(self._symbol, select)

    def check_market_watch(self) -> bool:
        """Checks if the symbol is selected in Market Watch and adds it to the Market Watch if necessary."""

        if not self.select():
            raise ValueError(f"Failed to select a symbol {self._symbol}")

        if not self.select(True):
            raise ValueError(f"Failed to add a symbol {self._symbol} in Market Watch")

        return True

    @property
    def is_synchronized(self) -> bool:
        """Checks the symbol synchronization with server."""
        return bool(self._info.select) if self._info else False

    # ------------- Volumes ---------------

    @property
    def volume(self) -> int:
        """Gets the volume of the last deal."""
        return int(self._tick_info["volume"])

    @property
    def volume_real(self) -> float:
        """Gets the real volume of the last deal."""
        return float(self._tick_info["volume_real"])

    @property
    def volume_high(self) -> int:
        """Gets the maximal volume for a day."""
        return int(self._info.volumehigh) if self._info else 0

    @property
    def volume_low(self) -> int:
        """Gets the minimal volume for a day."""
        return int(self._info.volumelow) if self._info else 0

    # ------------ Miscellaneous ------------

    @property
    def time(self) -> Optional[datetime]:
        """Gets the time of the last quote as UTC datetime."""
        tick_time = self._tick_info["time"]
        if tick_time is None:
            return None
        return datetime.fromtimestamp(tick_time, tz=timezone.utc)

    @property
    def time_msc(self) -> int:
        """Gets the time of the last quote in milliseconds."""
        return int(self._tick_info["time_msc"])

    @property
    def spread(self) -> float:
        """Gets the amount of spread in points."""
        return float(self._info.spread) if self._info else 0.0

    @property
    def spread_float(self) -> bool:
        """Gets the floating spread flag."""
        return bool(self._info.spread_float) if self._info else False

    @property
    def ticks_book_depth(self) -> int:
        """Gets the depth of ticks saving."""
        return int(self._info.ticks_bookdepth) if self._info else 0

    # -------------- Trade levels ---------------

    @property
    def stops_level(self) -> int:
        """Gets the minimal indent for orders in points."""
        return int(self._info.trade_stops_level) if self._info else 0

    @property
    def freeze_level(self) -> int:
        """Gets the freezing distance for trade operations in points."""
        return int(self._info.trade_freeze_level) if self._info else 0

    # ------------- Bid prices -------------

    @property
    def bid(self) -> float:
        """Returns the current bid price."""
        return float(self._tick_info["bid"])

    @property
    def bid_high(self) -> float:
        """Gets the maximal Bid price for a day."""
        return float(self._info.bidhigh) if self._info else 0.0

    @property
    def bid_low(self) -> float:
        """Gets the minimal Bid price for a day."""
        return float(self._info.bidlow) if self._info else 0.0

    # ------------- Ask prices -------------

    @property
    def ask(self) -> float:
        """Returns the current ask price."""
        return float(self._tick_info["ask"])

    @property
    def ask_high(self) -> float:
        """Gets the maximal Ask price for a day."""
        return float(self._info.askhigh) if self._info else 0.0

    @property
    def ask_low(self) -> float:
        """Gets the minimal Ask price for a day."""
        return float(self._info.asklow) if self._info else 0.0

    # ------------- Last prices -------------

    @property
    def last(self) -> float:
        """Gets the current Last price."""
        return float(self._tick_info["last"])

    @property
    def last_high(self) -> float:
        """Gets the maximal Last price for a day."""
        return float(self._info.lasthigh) if self._info else 0.0

    @property
    def last_low(self) -> float:
        """Gets the minimal Last price for a day."""
        return float(self._info.lastlow) if self._info else 0.0

    # ------------ Trade modes -------------

    @property
    def trade_calc_mode(self) -> int:
        """Contract price calculation mode"""
        return int(self._info.trade_calc_mode) if self._info else -1

    @property
    def trade_calc_mode_description(self) -> str:
        """A description of contract price calculation modes."""

        calc_mode_map = {
            self.terminal.SYMBOL_CALC_MODE_FOREX: "Calculation of profit and margin for Forex",
            self.terminal.SYMBOL_CALC_MODE_FUTURES: "Calculation of margin and profit for futures",
            self.terminal.SYMBOL_CALC_MODE_CFD: "Calculation of margin and profit for CFD",
            self.terminal.SYMBOL_CALC_MODE_CFDINDEX: "Calculation of margin and profit for CFD by indexes",
            self.terminal.SYMBOL_CALC_MODE_CFDLEVERAGE: "Calculation of margin and profit for CFD at leverage trading",
            self.terminal.SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE: "Calculation of profit and margin for Forex symbols without taking into account the leverage",
            self.terminal.SYMBOL_CALC_MODE_EXCH_STOCKS: "Calculation of margin and profit for trading securities on a stock exchange",
            self.terminal.SYMBOL_CALC_MODE_EXCH_FUTURES: "Calculation of margin and profit for trading futures contracts on a stock exchange",
            self.terminal.SYMBOL_CALC_MODE_EXCH_BONDS: "Calculation of margin and profit for trading bonds on a stock exchange",
            self.terminal.SYMBOL_CALC_MODE_EXCH_STOCKS_MOEX: "Calculation of margin and profit for trading securities on MOEX",
            self.terminal.SYMBOL_CALC_MODE_EXCH_BONDS_MOEX: "Calculation of margin and profit for trading bonds on MOEX",
            self.terminal.SYMBOL_CALC_MODE_SERV_COLLATERAL: "Collateral mode - a symbol is used as a non-tradable asset on a trading account"
        }
        return calc_mode_map.get(self.trade_calc_mode, "Unknown trade calculation mode")

    @property
    def trade_mode(self) -> int:
        """Symbol trading modes (restrictions)"""
        return int(self._info.trade_mode) if self._info else -1

    @property
    def trade_mode_description(self) -> str:
        """A description of symbol trading modes (restrictions)"""
        trade_mode_map = {
            self.terminal.SYMBOL_TRADE_MODE_DISABLED: "Trade is disabled for the symbol",
            self.terminal.SYMBOL_TRADE_MODE_LONGONLY: "Allowed only long positions",
            self.terminal.SYMBOL_TRADE_MODE_SHORTONLY: "Allowed only short positions",
            self.terminal.SYMBOL_TRADE_MODE_CLOSEONLY: "Allowed only position close operations",
            self.terminal.SYMBOL_TRADE_MODE_FULL: "No trade restrictions"
        }
        return trade_mode_map.get(self.trade_mode, "Unknown trade mode")

    @property
    def trade_execution(self) -> int:
        """Possible deal execution modes for a given symbol."""
        return int(self._info.trade_exemode) if self._info else -1

    @property
    def trade_execution_description(self) -> str:
        """A description of possible deal execution modes for a given symbol."""

        exec_mode_map = {
            self.terminal.SYMBOL_TRADE_EXECUTION_REQUEST: "Execution by request",
            self.terminal.SYMBOL_TRADE_EXECUTION_INSTANT: "Instant execution",
            self.terminal.SYMBOL_TRADE_EXECUTION_MARKET: "Market execution",
            self.terminal.SYMBOL_TRADE_EXECUTION_EXCHANGE: "Exchange execution"
        }
        return exec_mode_map.get(self.trade_execution, "Unkown trade execution mode")

    @property
    def order_mode(self) -> int:
        """Flags of allowed order types for the symbol (bit field)."""
        return int(self._info.order_mode) if self._info else 0

    # ----------- Swaps -------------

    @property
    def swap_mode(self) -> int:
        """A method of swap calculation at position transfer."""
        return int(self._info.swap_mode) if self._info else -1

    @property
    def swap_mode_description(self) -> str:
        """A description of methods of swap calculation at position transfer."""

        swap_mode_map = {
            self.terminal.SYMBOL_SWAP_MODE_DISABLED: "No swaps",
            self.terminal.SYMBOL_SWAP_MODE_POINTS: "Swaps are calculated in points",
            self.terminal.SYMBOL_SWAP_MODE_CURRENCY_SYMBOL: "Swaps are calculated in base currency",
            self.terminal.SYMBOL_SWAP_MODE_CURRENCY_MARGIN: "Swaps are calculated in margin currency",
            self.terminal.SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT: "Swaps are calculated in deposit currency",
            self.terminal.SYMBOL_SWAP_MODE_INTEREST_CURRENT: "Swaps are calculated as annual interest using the current price",
            self.terminal.SYMBOL_SWAP_MODE_INTEREST_OPEN: "Swaps are calculated as annual interest using the open price",
            self.terminal.SYMBOL_SWAP_MODE_REOPEN_CURRENT: "Swaps are charged by reopening positions at the close price",
            self.terminal.SYMBOL_SWAP_MODE_REOPEN_BID: "Swaps are charged by reopening positions at the Bid price"
        }
        return swap_mode_map.get(self.swap_mode, "Unkown swap mode")

    @property
    def swap_rollover_3days(self) -> int:
        """The day of week to charge 3-day swap rollover"""
        return int(self._info.swap_rollover3days) if self._info else -1

    @property
    def swap_rollover_3days_description(self) -> str:
        """A description (string) of the day of week to charge 3-day swap rollover"""

        swap_rollover_map = {
            0: "Sunday",
            1: "Monday",
            2: "Tuesday",
            3: "Wednesday",
            4: "Thursday",
            5: "Friday",
            6: "Saturday",
        }
        return swap_rollover_map.get(self.swap_rollover_3days, "Unkown swap rollover 3 days")

    @property
    def filling_mode(self) -> int:
        """Flags of allowed order filling modes for the symbol (bit field)."""
        return int(self._info.filling_mode) if self._info else 0

    # ------------- Dates for futures ----------------

    @property
    def expiration_time(self) -> Optional[datetime]:
        """Date of the symbol trade end (usually used for futures)"""
        if not self._info or int(self._info.expiration_time) == 0:
            return None
        return datetime.fromtimestamp(self._info.expiration_time, tz=timezone.utc)

    @property
    def start_time(self) -> Optional[datetime]:
        """Date of the symbol trade beginning (usually used for futures)"""
        if not self._info or int(self._info.start_time) == 0:
            return None
        return datetime.fromtimestamp(self._info.start_time, tz=timezone.utc)

    # -----------  Margin parameters -------------

    @property
    def margin_initial(self) -> float:
        """
            Initial margin means the amount in the margin currency required for opening a position with the volume of one lot. It is used for checking a client's assets when he or she enters the market.

            The SymbolInfoMarginRate() function provides data on the amount of charged margin depending on the order type and direction.
        """
        return float(self._info.margin_initial) if self._info else 0.0

    @property
    def margin_maintenance(self) -> float:
        """
            The maintenance margin. If it is set, it sets the margin amount in the margin currency of the symbol, charged from one lot. It is used for checking a client's assets when his/her account state changes. If the maintenance margin is equal to 0, the initial margin is used.

            The SymbolInfoMarginRate() function provides data on the amount of charged margin depending on the order type and direction.
        """
        return float(self._info.margin_maintenance) if self._info else 0.0

    @property
    def margin_hedged(self) -> float:

        """
            Contract size or margin value per one lot of hedged positions (oppositely directed positions of one symbol). Two margin calculation methods are possible for hedged positions. The calculation method is defined by the broker.

            Basic calculation:
                - If the initial margin (margin_initial) is specified for a symbol, the hedged margin is specified as an absolute value (in monetary terms).
                - If the initial margin is not specified (equal to 0), SYMBOL_MARGIN_HEDGED is equal to the size of the contract, that will be used to calculate the margin by the appropriate formula in accordance with the type of the financial instrument (trade_calc_mode).

            Calculation for the largest position:
                - The SYMBOL_MARGIN_HEDGED value is not taken into account.
                - The volume of all short and all long positions of a symbol is calculated.
                - For each direction, a weighted average open price and a weighted average rate of conversion to the deposit currency is calculated.
                - Next, using the appropriate formula chosen in accordance with the symbol type (trade_calc_mode) the margin is calculated for the short and the long part.
                - The largest one of the values is used as the margin.
        """

        return float(self._info.margin_hedged) if self._info else 0.0

    @property
    def margin_hedged_use_leg(self) -> bool:
        """
            Boolean flag to enable (true) or disable (false) the hedged margin calculation mode for the largest of the overlapped positions (buy and sell
            positions of the same symbol).

            If the flag is enabled, the margin for the largest position is calculated using the SYMBOL_MARGIN_HEDGED value. If the flag is disabled, the margin for the largest position is calculated using the SYMBOL_MARGIN_INITIAL value.
        """

        return bool(self._info.margin_hedged_use_leg) if self._info else False

    # -------------------------
    # Tick parameters / quantization
    # -------------------------

    @property
    def digits(self) -> int:
        """Digits after a decimal point for the symbol prices."""
        return int(self._info.digits) if self._info else 0

    @property
    def point(self) -> float:
        """Symbol point value/The smallest price change for the symbol."""
        return float(self._info.point) if self._info else 0.0

    @property
    def tick_value(self) -> float:
        """The value of a single tick for the symbol."""
        return float(self._info.trade_tick_value) if self._info else 0.0

    @property
    def tick_value_profit(self) -> float:
        """Calculated tick price for a profitable position"""
        return float(self._info.trade_tick_value_profit) if self._info else 0.0

    @property
    def tick_value_loss(self) -> float:
        """Calculated tick price for a losing position"""
        return float(self._info.trade_tick_value_loss) if self._info else 0.0

    @property
    def tick_size(self) -> float:
        """Minimal price change / The size of a single tick for the symbol."""
        return float(self._info.trade_tick_size) if self._info else 0.0

    @property
    def swap_long(self) -> float:
        """Gets the swap for long positions."""
        return float(self._info.swap_long) if self._info else 0.0

    @property
    def swap_short(self) -> float:
        """Gets the swap for short positions."""
        return float(self._info.swap_short) if self._info else 0.0

    # ------------ Contract / lots parameters ------------

    @property
    def contract_size(self) -> float:
        """Trade contract size"""
        return float(self._info.trade_contract_size) if self._info else 0.0

    @property
    def lots_min(self) -> float:
        """Minimum allowed volume of a position in lots."""
        return float(self._info.volume_min) if self._info else 0.0

    @property
    def lots_max(self) -> float:
        """Maximum allowed volume of a position in lots."""
        return float(self._info.volume_max) if self._info else 0.0

    @property
    def lots_step(self) -> float:
        """Minimal volume change step for deal executions."""
        return float(self._info.volume_step) if self._info else 0.0

    @property
    def lots_limit(self) -> float:
        """
            Maximum allowed aggregate volume of an open position and pending orders in one direction (buy or sell) for the symbol. For example, with the limitation of 5 lots, you can have an open buy position with the volume of 5 lots and place a pending order Sell Limit with the volume of 5 lots. But in this case you cannot place a Buy Limit pending order (since the total volume in one direction will exceed the limitation) or place Sell Limit with the volume more than 5 lots.
        """

        return float(self._info.volume_limit) if self._info else 0.0

    # -------------- Currency / text properties ------------

    @property
    def currency_base(self) -> str:
        """Base currency of the symbol."""
        return str(self._info.currency_base) if self._info else ""

    @property
    def currency_profit(self) -> str:
        """Profit currency for the symbol"""
        return str(self._info.currency_profit) if self._info else ""

    @property
    def currency_margin(self) -> str:
        """Margin currency for the symbol"""
        return str(self._info.currency_margin) if self._info else ""

    @property
    def bank(self) -> str:
        """Feeder of the current quote"""
        return str(self._info.bank) if self._info else ""

    @property
    def description(self) -> str:
        """Symbol description"""
        return str(self._info.description) if self._info else ""

    @property
    def path(self) -> str:
        """Path to the symbol tree. For example, "Forex\\EURUSD" or "CFD\\Apple Inc."""
        return str(self._info.path) if self._info else ""

    @property
    def page(self) -> str:
        """
        The address of the web page containing symbol information.

        This address will be displayed as a link when viewing symbol properties in the terminal
        """

        return str(self._info.page) if self._info else ""

    # -------------------------
    # Session properties
    # -------------------------

    @property
    def session_deals(self) -> int:
        """Number of deals in the current session."""
        return int(self._info.session_deals) if self._info else 0

    @property
    def session_buy_orders(self) -> int:
        """Number of buy orders at the moment (in the current session)."""
        return int(self._info.session_buy_orders) if self._info else 0

    @property
    def session_sell_orders(self) -> int:
        """Number of sell orders at the moment (in the current session)."""
        return int(self._info.session_sell_orders) if self._info else 0

    @property
    def session_turnover(self) -> float:
        """Summary turnover of the current session"""
        return float(self._info.session_turnover) if self._info else 0.0

    @property
    def session_interest(self) -> float:
        """Summary open interest"""
        return float(self._info.session_interest) if self._info else 0.0

    @property
    def session_buy_orders_volume(self) -> float:
        """Current volume of Buy orders"""
        return float(self._info.session_buy_orders_volume) if self._info else 0.0

    @property
    def session_sell_orders_volume(self) -> float:
        """Current volume of Sell orders"""
        return float(self._info.session_sell_orders_volume) if self._info else 0.0

    @property
    def session_open(self) -> float:
        """Opening price of the current session"""
        return float(self._info.session_open) if self._info else 0.0

    @property
    def session_close(self) -> float:
        """Closing price of the current session"""
        return float(self._info.session_close) if self._info else 0.0

    @property
    def session_aw(self) -> float:
        """Average weighted price of the current session"""
        return float(self._info.session_aw) if self._info else 0.0

    @property
    def session_price_settlement(self) -> float:
        """Settlement price of the current session"""
        return float(self._info.session_price_settlement) if self._info else 0.0

    @property
    def session_price_limit_min(self) -> float:
        """Minimal price for the current session"""
        return float(self._info.session_price_limit_min) if self._info else 0.0

    @property
    def session_price_limit_max(self) -> float:
        """Maximal price for the current session"""
        return float(self._info.session_price_limit_max) if self._info else 0.0

    # -------------- Generic Info* methods ------------

    def info_integer(self, prop_name: str) -> Optional[int]:
        """
        Gets the value of a specified integer type property.

        Args:
            prop_name : Name of the symbol info attribute.

        Returns:
            Integer value if present, otherwise None.
        """
        if self._info is None or not hasattr(self._info, prop_name):
            return None
        value = getattr(self._info, prop_name)
        return None if value is None else int(value)

    def info_double(self, prop_name: str) -> Optional[float]:
        """
        Gets the value of a specified double type property.

        Args:
            Name of the symbol info attribute.

        Returns:
            Float value if present, otherwise None.
        """
        if self._info is None or not hasattr(self._info, prop_name):
            return None
        value = getattr(self._info, prop_name)

        return None if value is None else float(value)

    def info_string(self, prop_name: str) -> Optional[str]:
        """
        Gets the value of a specified string type property.

        Args:
            prop_name : Name of the symbol info attribute.

        Returns:
            String value if present, otherwise None.
        """
        if self._info is None or not hasattr(self._info, prop_name):
            return None
        value = getattr(self._info, prop_name)
        return None if value is None else str(value)

    # --------------Service functions --------------

    def normalize_price(self, price: float) -> float:
        """
        Returns the value of price normalized using the symbol properties.

        Args:
            price : Price to normalize.

        Returns:
            Normalized price based on symbol digits.
        """
        return round(float(price), self.digits)

    # ----------- Debug / utility helpers ---------------

    def to_dict(self) -> dict[str, Any]:
        """Return all @property values as a dictionary."""
        data: dict[str, Any] = {}
        seen: set[str] = set()

        for cls in type(self).mro():
            for name, attr in vars(cls).items():
                if name in seen:
                    continue
                if isinstance(attr, property):
                    seen.add(name)
                    try:
                        data[name] = getattr(self, name)
                    except Exception as e:
                        data[name] = f"<error: {e}>"

        return data

    def print_all(self) -> None:
        """Print all @property values of the class."""
        for name, value in self.to_dict().items():
            print(f"{name:28} : {value}")

    def __repr__(self) -> str:
        props = self.to_dict()
        props_str = ", ".join(f"{k}={v!r}" for k, v in props.items())
        return f"{self.__class__.__name__}({props_str})"