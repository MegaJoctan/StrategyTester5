from dataclasses import dataclass


@dataclass
class MetaTrader5Constants:
    """MetaTrader 5 constants, such as timeframes, order types, deal types, etc."""
    # timeframes
    TIMEFRAME_M1 = 1
    """One-minute timeframe."""
    TIMEFRAME_M2 = 2
    """Two-minute timeframe."""
    TIMEFRAME_M3 = 3
    """Three-minute timeframe."""
    TIMEFRAME_M4 = 4
    """Four-minute timeframe."""
    TIMEFRAME_M5 = 5
    """Five-minute timeframe."""
    TIMEFRAME_M6 = 6
    """Six-minute timeframe."""
    TIMEFRAME_M10 = 10
    """Ten-minute timeframe."""
    TIMEFRAME_M12 = 12
    """Twelve-minute timeframe."""
    TIMEFRAME_M15 = 15
    """Fifteen-minute timeframe."""
    TIMEFRAME_M20 = 20
    """Twenty-minute timeframe."""
    TIMEFRAME_M30 = 30
    """Thirty-minute timeframe."""
    TIMEFRAME_H1 = 1 | 0x4000
    """One-hour timeframe."""
    TIMEFRAME_H2 = 2 | 0x4000
    """Two-hour timeframe."""
    TIMEFRAME_H4 = 4 | 0x4000
    """Four-hour timeframe."""
    TIMEFRAME_H3 = 3 | 0x4000
    """Three-hour timeframe."""
    TIMEFRAME_H6 = 6 | 0x4000
    """Six-hour timeframe."""
    TIMEFRAME_H8 = 8 | 0x4000
    """Eight-hour timeframe."""
    TIMEFRAME_H12 = 12 | 0x4000
    """Twelve-hour timeframe."""
    TIMEFRAME_D1 = 24 | 0x4000
    """One-day timeframe."""
    TIMEFRAME_W1 = 1 | 0x8000
    """One-week timeframe."""
    TIMEFRAME_MN1 = 1 | 0xC000
    """One-Month timeframe."""
    # tick copy flags
    COPY_TICKS_ALL = -1
    COPY_TICKS_INFO = 1
    COPY_TICKS_TRADE = 2
    # tick flags
    TICK_FLAG_BID = 0x02
    TICK_FLAG_ASK = 0x04
    TICK_FLAG_LAST = 0x08
    TICK_FLAG_VOLUME = 0x10
    TICK_FLAG_BUY = 0x20
    TICK_FLAG_SELL = 0x40

    # position type, ENUM_POSITION_TYPE
    POSITION_TYPE_BUY = 0
    """Buy"""
    POSITION_TYPE_SELL = 1
    """Sell"""
    # position reason, ENUM_POSITION_REASON
    POSITION_REASON_CLIENT = 0
    """The position was opened as a result of activation of an order placed from a desktop terminal"""
    POSITION_REASON_MOBILE = 1
    """The position was opened as a result of activation of an order placed from a mobile application"""
    POSITION_REASON_WEB = 2
    """The position was opened as a result of activation of an order placed from the web platform"""
    POSITION_REASON_EXPERT = 3
    """The position was opened as a result of activation of an order placed from an MQL5 program, i.e. an Expert Advisor or a script"""

    # order types, ENUM_ORDER_TYPE

    ORDER_TYPE_BUY = 0
    """Market Buy order"""
    ORDER_TYPE_SELL = 1
    """Market Sell order"""
    ORDER_TYPE_BUY_LIMIT = 2
    """Buy Limit pending order"""
    ORDER_TYPE_SELL_LIMIT = 3
    """Sell Limit pending order"""
    ORDER_TYPE_BUY_STOP = 4
    """Buy Stop pending order"""
    ORDER_TYPE_SELL_STOP = 5
    """Sell Stop pending order"""
    ORDER_TYPE_BUY_STOP_LIMIT = 6
    """Upon reaching the order price, a pending Buy Limit order is placed at the StopLimit price"""
    ORDER_TYPE_SELL_STOP_LIMIT = 7
    """Upon reaching the order price, a pending Sell Limit order is placed at the StopLimit price"""
    ORDER_TYPE_CLOSE_BY = 8
    """Order to close a position by an opposite one"""
    # order state, ENUM_ORDER_STATE
    ORDER_STATE_STARTED = 0
    """Order checked, but not yet accepted by broker"""
    ORDER_STATE_PLACED = 1
    """Order accepted"""
    ORDER_STATE_CANCELED = 2
    """Order canceled by client"""
    ORDER_STATE_PARTIAL = 3
    """Order partially executed"""
    ORDER_STATE_FILLED = 4
    """Order fully executed"""
    ORDER_STATE_REJECTED = 5
    """Order rejected"""
    ORDER_STATE_EXPIRED = 6
    """Order expired"""
    ORDER_STATE_REQUEST_ADD = 7
    """Order is being registered (placing to the trading system)"""
    ORDER_STATE_REQUEST_MODIFY = 8
    """Order is being modified (changing its parameters)"""
    ORDER_STATE_REQUEST_CANCEL = 9
    """Order is being deleted (deleting from the trading system)"""

    # ENUM_ORDER_TYPE_FILLING

    ORDER_FILLING_FOK = 0
    """Fill Or Kill order"""
    ORDER_FILLING_IOC = 1
    """Immediately Or Cancel"""
    ORDER_FILLING_RETURN = 2
    """Return remaining volume to book"""
    ORDER_FILLING_BOC = 3
    """Book Or Cancel order"""

    # ENUM_ORDER_TYPE_TIME

    ORDER_TIME_GTC: int = 0  # Good till cancel order
    """ORDER_TIME_GTC - Good till cancel order"""
    ORDER_TIME_DAY = 1
    """ Good till current trade day order"""
    ORDER_TIME_SPECIFIED = 2
    """ Good till expired order"""
    ORDER_TIME_SPECIFIED_DAY = 3
    """ The order will be effective till 23:59:59 of the specified day. If this time is outside a trading session, the order expires in the nearest trading time."""
    # ENUM_ORDER_REASON
    ORDER_REASON_CLIENT = 0
    """The order was placed from a desktop terminal"""
    ORDER_REASON_MOBILE = 1
    """The order was placed from a mobile application"""
    ORDER_REASON_WEB = 2
    """The order was placed from a web platform"""
    ORDER_REASON_EXPERT = 3
    """The order was placed from an MQL5-program, i.e. by an Expert Advisor or a script"""
    ORDER_REASON_SL = 4
    """The order was placed as a result of Stop Loss activation"""
    ORDER_REASON_TP = 5
    """The order was placed as a result of Take Profit activation"""
    ORDER_REASON_SO = 6
    """The order was placed as a result of the Stop Out event"""
    # deal types, ENUM_DEAL_TYPE
    DEAL_TYPE_BUY = 0
    """Buy"""
    DEAL_TYPE_SELL = 1
    """Sell"""
    DEAL_TYPE_BALANCE = 2
    """Balance"""
    DEAL_TYPE_CREDIT = 3
    """Credit"""
    DEAL_TYPE_CHARGE = 4
    """Additional charge"""
    DEAL_TYPE_CORRECTION = 5
    """Correction"""
    DEAL_TYPE_BONUS = 6
    """Bonus"""
    DEAL_TYPE_COMMISSION = 7
    """Additional commission"""
    DEAL_TYPE_COMMISSION_DAILY = 8
    """Daily commission"""
    DEAL_TYPE_COMMISSION_MONTHLY = 9
    """Monthly commission"""
    DEAL_TYPE_COMMISSION_AGENT_DAILY = 10
    """Daily agent commission"""
    DEAL_TYPE_COMMISSION_AGENT_MONTHLY = 11
    """Monthly agent commission"""
    DEAL_TYPE_INTEREST = 12
    """Interest rate"""
    DEAL_TYPE_BUY_CANCELED = 13
    """Canceled buy deal."""
    DEAL_TYPE_SELL_CANCELED = 14
    """Canceled sell deal."""
    DEAL_DIVIDEND = 15
    """Dividend operations"""
    DEAL_DIVIDEND_FRANKED = 16
    """Franked (non-taxable) dividend operations"""
    DEAL_TAX = 17
    """Tax charges"""
    # ENUM_DEAL_ENTRY
    DEAL_ENTRY_IN = 0
    """Entry in"""
    DEAL_ENTRY_OUT = 1
    """Entry out"""
    DEAL_ENTRY_INOUT = 2
    """Reverse"""
    DEAL_ENTRY_OUT_BY = 3
    """Close a position by an opposite one"""
    # ENUM_DEAL_REASON
    DEAL_REASON_CLIENT = 0
    """The deal was executed as a result of activation of an order placed from a desktop terminal"""
    DEAL_REASON_MOBILE = 1
    """The deal was executed as a result of activation of an order placed from a mobile application"""
    DEAL_REASON_WEB = 2
    """The deal was executed as a result of activation of an order placed from the web platform"""
    DEAL_REASON_EXPERT = 3
    """The deal was executed as a result of activation of an order placed from an MQL5 program, i.e. an Expert Advisor or a script"""
    DEAL_REASON_SL = 4
    """The deal was executed as a result of Stop Loss activation"""
    DEAL_REASON_TP = 5
    """The deal was executed as a result of Take Profit activation"""
    DEAL_REASON_SO = 6
    """The deal was executed as a result of the Stop Out event"""
    DEAL_REASON_ROLLOVER = 7
    """The deal was executed due to a rollover"""
    DEAL_REASON_VMARGIN = 8
    """The deal was executed after charging the variation margin"""
    DEAL_REASON_SPLIT = 9
    """The deal was executed after the split (price reduction) of an instrument, which had an open position during split announcement"""
    # ENUM_TRADE_REQUEST_ACTIONS, Trade Operation Types
    TRADE_ACTION_DEAL = 1
    """Place a trade order for an immediate execution with the specified parameters (market order)"""
    TRADE_ACTION_PENDING = 5
    """Place a trade order for the execution under specified conditions (pending order)"""
    TRADE_ACTION_SLTP = 6
    """Modify Stop Loss and Take Profit values of an opened position"""
    TRADE_ACTION_MODIFY = 7
    """Modify the parameters of the order placed previously"""
    TRADE_ACTION_REMOVE = 8
    """Delete the pending order placed previously"""
    TRADE_ACTION_CLOSE_BY = 10
    """Close a position by an opposite one"""
    # ENUM_SYMBOL_CHART_MODE
    SYMBOL_CHART_MODE_BID = 0
    SYMBOL_CHART_MODE_LAST = 1
    # ENUM_SYMBOL_CALC_MODE
    SYMBOL_CALC_MODE_FOREX = 0
    SYMBOL_CALC_MODE_FUTURES = 1
    SYMBOL_CALC_MODE_CFD = 2
    SYMBOL_CALC_MODE_CFDINDEX = 3
    SYMBOL_CALC_MODE_CFDLEVERAGE = 4
    SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE = 5
    SYMBOL_CALC_MODE_EXCH_STOCKS = 32
    SYMBOL_CALC_MODE_EXCH_FUTURES = 33
    SYMBOL_CALC_MODE_EXCH_OPTIONS = 34
    SYMBOL_CALC_MODE_EXCH_OPTIONS_MARGIN = 36
    SYMBOL_CALC_MODE_EXCH_BONDS = 37
    SYMBOL_CALC_MODE_EXCH_STOCKS_MOEX = 38
    SYMBOL_CALC_MODE_EXCH_BONDS_MOEX = 39
    SYMBOL_CALC_MODE_SERV_COLLATERAL = 64
    # ENUM_SYMBOL_TRADE_MODE
    SYMBOL_TRADE_MODE_DISABLED = 0
    SYMBOL_TRADE_MODE_LONGONLY = 1
    SYMBOL_TRADE_MODE_SHORTONLY = 2
    SYMBOL_TRADE_MODE_CLOSEONLY = 3
    SYMBOL_TRADE_MODE_FULL = 4
    # ENUM_SYMBOL_TRADE_EXECUTION
    SYMBOL_TRADE_EXECUTION_REQUEST = 0
    SYMBOL_TRADE_EXECUTION_INSTANT = 1
    SYMBOL_TRADE_EXECUTION_MARKET = 2
    SYMBOL_TRADE_EXECUTION_EXCHANGE = 3
    # ENUM_SYMBOL_SWAP_MODE
    SYMBOL_SWAP_MODE_DISABLED = 0
    SYMBOL_SWAP_MODE_POINTS = 1
    SYMBOL_SWAP_MODE_CURRENCY_SYMBOL = 2
    SYMBOL_SWAP_MODE_CURRENCY_MARGIN = 3
    SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT = 4
    SYMBOL_SWAP_MODE_INTEREST_CURRENT = 5
    SYMBOL_SWAP_MODE_INTEREST_OPEN = 6
    SYMBOL_SWAP_MODE_REOPEN_CURRENT = 7
    SYMBOL_SWAP_MODE_REOPEN_BID = 8
    # ENUM_DAY_OF_WEEK
    DAY_OF_WEEK_SUNDAY = 0
    DAY_OF_WEEK_MONDAY = 1
    DAY_OF_WEEK_TUESDAY = 2
    DAY_OF_WEEK_WEDNESDAY = 3
    DAY_OF_WEEK_THURSDAY = 4
    DAY_OF_WEEK_FRIDAY = 5
    DAY_OF_WEEK_SATURDAY = 6
    # ENUM_SYMBOL_ORDER_GTC_MODE
    SYMBOL_ORDERS_GTC = 0
    SYMBOL_ORDERS_DAILY = 1
    SYMBOL_ORDERS_DAILY_NO_STOPS = 2
    # ENUM_SYMBOL_OPTION_RIGHT
    SYMBOL_OPTION_RIGHT_CALL = 0
    SYMBOL_OPTION_RIGHT_PUT = 1
    # ENUM_SYMBOL_OPTION_MODE
    SYMBOL_OPTION_MODE_EUROPEAN = 0
    SYMBOL_OPTION_MODE_AMERICAN = 1
    # ENUM_ACCOUNT_TRADE_MODE
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_CONTEST = 1
    ACCOUNT_TRADE_MODE_REAL = 2
    # ENUM_ACCOUNT_STOPOUT_MODE
    ACCOUNT_STOPOUT_MODE_PERCENT = 0
    ACCOUNT_STOPOUT_MODE_MONEY = 1
    # ENUM_ACCOUNT_MARGIN_MODE
    ACCOUNT_MARGIN_MODE_RETAIL_NETTING = 0
    ACCOUNT_MARGIN_MODE_EXCHANGE = 1
    ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 2
    # ENUM_BOOK_TYPE
    BOOK_TYPE_SELL = 1
    BOOK_TYPE_BUY = 2
    BOOK_TYPE_SELL_MARKET = 3
    BOOK_TYPE_BUY_MARKET = 4
    # order send/check return codes
    TRADE_RETCODE_REQUOTE = 10004
    TRADE_RETCODE_REJECT = 10006
    TRADE_RETCODE_CANCEL = 10007
    TRADE_RETCODE_PLACED = 10008
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_DONE_PARTIAL = 10010
    TRADE_RETCODE_ERROR = 10011
    TRADE_RETCODE_TIMEOUT = 10012
    TRADE_RETCODE_INVALID = 10013
    TRADE_RETCODE_INVALID_VOLUME = 10014
    TRADE_RETCODE_INVALID_PRICE = 10015
    TRADE_RETCODE_INVALID_STOPS = 10016
    TRADE_RETCODE_TRADE_DISABLED = 10017
    TRADE_RETCODE_MARKET_CLOSED = 10018
    TRADE_RETCODE_NO_MONEY = 10019
    TRADE_RETCODE_PRICE_CHANGED = 10020
    TRADE_RETCODE_PRICE_OFF = 10021
    TRADE_RETCODE_INVALID_EXPIRATION = 10022
    TRADE_RETCODE_ORDER_CHANGED = 10023
    TRADE_RETCODE_TOO_MANY_REQUESTS = 10024
    TRADE_RETCODE_NO_CHANGES = 10025
    TRADE_RETCODE_SERVER_DISABLES_AT = 10026
    TRADE_RETCODE_CLIENT_DISABLES_AT = 10027
    TRADE_RETCODE_LOCKED = 10028
    TRADE_RETCODE_FROZEN = 10029
    TRADE_RETCODE_INVALID_FILL = 10030
    TRADE_RETCODE_CONNECTION = 10031
    TRADE_RETCODE_ONLY_REAL = 10032
    TRADE_RETCODE_LIMIT_ORDERS = 10033
    TRADE_RETCODE_LIMIT_VOLUME = 10034
    TRADE_RETCODE_INVALID_ORDER = 10035
    TRADE_RETCODE_POSITION_CLOSED = 10036
    TRADE_RETCODE_INVALID_CLOSE_VOLUME = 10038
    TRADE_RETCODE_CLOSE_ORDER_EXIST = 10039
    TRADE_RETCODE_LIMIT_POSITIONS = 10040
    TRADE_RETCODE_REJECT_CANCEL = 10041
    TRADE_RETCODE_LONG_ONLY = 10042
    TRADE_RETCODE_SHORT_ONLY = 10043
    TRADE_RETCODE_CLOSE_ONLY = 10044
    TRADE_RETCODE_FIFO_CLOSE = 10045
    # functio error codes, last_error()
    RES_S_OK = 1  # generic success
    RES_E_FAIL = -1  # generic fail
    RES_E_INVALID_PARAMS = -2  # invalid arguments/parameters
    RES_E_NO_MEMORY = -3  # no memory condition
    RES_E_NOT_FOUND = -4  # no history
    RES_E_INVALID_VERSION = -5  # invalid version
    RES_E_AUTH_FAILED = -6  # authorization failed
    RES_E_UNSUPPORTED = -7  # unsupported method
    RES_E_AUTO_TRADING_DISABLED = -8  # auto-trading disabled
    RES_E_INTERNAL_FAIL = -10000  # internal IPC general error
    RES_E_INTERNAL_FAIL_SEND = -10001  # internal IPC send failed
    RES_E_INTERNAL_FAIL_RECEIVE = -10002  # internal IPC recv failed
    RES_E_INTERNAL_FAIL_INIT = -10003  # internal IPC initialization fail
    RES_E_INTERNAL_FAIL_CONNECT = -10004  # internal IPC no ipc
    RES_E_INTERNAL_FAIL_TIMEOUT = -10005  # internal timeout

    RETCODE_MAP = {
        TRADE_RETCODE_REQUOTE: "Requote",
        TRADE_RETCODE_REJECT: "Request rejected",
        TRADE_RETCODE_CANCEL: "Request canceled by trader",
        TRADE_RETCODE_PLACED: "Order placed",
        TRADE_RETCODE_DONE: "Request completed",
        TRADE_RETCODE_DONE_PARTIAL: "Only part of the request was completed",
        TRADE_RETCODE_ERROR: "Request processing error",
        TRADE_RETCODE_TIMEOUT: "Request canceled by timeout",
        TRADE_RETCODE_INVALID: "Invalid request",
        TRADE_RETCODE_INVALID_VOLUME: "Invalid volume in the request",
        TRADE_RETCODE_INVALID_PRICE: "Invalid price in the request",
        TRADE_RETCODE_INVALID_STOPS: "Invalid stops in the request",
        TRADE_RETCODE_TRADE_DISABLED: "Trade is disabled",
        TRADE_RETCODE_MARKET_CLOSED: "Market is closed",
        TRADE_RETCODE_NO_MONEY: "There is not enough money to complete the request",
        TRADE_RETCODE_PRICE_CHANGED: "Prices changed",
        TRADE_RETCODE_PRICE_OFF: "There are no quotes to process the request",
        TRADE_RETCODE_INVALID_EXPIRATION: "Invalid order expiration date in the request",
        TRADE_RETCODE_ORDER_CHANGED: "Order state changed",
        TRADE_RETCODE_TOO_MANY_REQUESTS: "Too frequent requests",
        TRADE_RETCODE_NO_CHANGES: "No changes in request",
        TRADE_RETCODE_SERVER_DISABLES_AT: "Autotrading disabled by server",
        TRADE_RETCODE_CLIENT_DISABLES_AT: "Autotrading disabled by client terminal",
        TRADE_RETCODE_LOCKED: "Request locked for processing",
        TRADE_RETCODE_FROZEN: "Order or position frozen",
        TRADE_RETCODE_INVALID_FILL: "Invalid order filling type",
        TRADE_RETCODE_CONNECTION: "No connection with the trade server",
        TRADE_RETCODE_ONLY_REAL: "Operation is allowed only for live accounts",
        TRADE_RETCODE_LIMIT_ORDERS: "The number of pending orders has reached the limit",
        TRADE_RETCODE_LIMIT_VOLUME: "The volume of orders and positions for the symbol has reached the limit",
        TRADE_RETCODE_INVALID_ORDER: "Invalid order",
        TRADE_RETCODE_POSITION_CLOSED: "Position is closed",
        TRADE_RETCODE_INVALID_CLOSE_VOLUME: "Invalid close volume",
        TRADE_RETCODE_CLOSE_ORDER_EXIST: "A close order already exists",
        TRADE_RETCODE_LIMIT_POSITIONS: "The number of positions has reached the limit",
        TRADE_RETCODE_REJECT_CANCEL: "Order cancel request rejected",
        TRADE_RETCODE_LONG_ONLY: "Symbol supports only long positions",
        TRADE_RETCODE_SHORT_ONLY: "Symbol supports only short positions",
        TRADE_RETCODE_CLOSE_ONLY: "Only closing of positions is allowed",
        TRADE_RETCODE_FIFO_CLOSE: "FIFO close restriction applies",
    }

    DEAL_TYPE_MAP = {
        DEAL_TYPE_BUY: "BUY",
        DEAL_TYPE_SELL: "SELL",
        DEAL_TYPE_BALANCE: "BALANCE",
        DEAL_TYPE_CREDIT: "CREDIT",
        DEAL_TYPE_CHARGE: "CHARGE",
        DEAL_TYPE_CORRECTION: "CORRECTION",
        DEAL_TYPE_BONUS: "BONUS",
        DEAL_TYPE_COMMISSION: "COMMISSION",
        DEAL_TYPE_COMMISSION_DAILY: "COMMISSION DAILY",
        DEAL_TYPE_COMMISSION_MONTHLY: "COMMISSION MONTHLY",
        DEAL_TYPE_COMMISSION_AGENT_DAILY: "AGENT COMMISSION DAILY",
        DEAL_TYPE_COMMISSION_AGENT_MONTHLY: "AGENT COMMISSION MONTHLY",
        DEAL_TYPE_INTEREST: "INTEREST",
        DEAL_TYPE_BUY_CANCELED: "BUY CANCELED",
        DEAL_TYPE_SELL_CANCELED: "SELL CANCELED"
    }

    DEAL_ENTRY_MAP = {
        DEAL_ENTRY_IN: "IN",
        DEAL_ENTRY_OUT: "OUT",
        DEAL_ENTRY_INOUT: "INOUT"
    }

    ORDER_TYPE_MAP = {
        ORDER_TYPE_BUY: "Market Buy order",
        ORDER_TYPE_SELL: "Market Sell order",
        ORDER_TYPE_BUY_LIMIT: "Buy Limit pending order",
        ORDER_TYPE_SELL_LIMIT: "Sell Limit pending order",
        ORDER_TYPE_BUY_STOP: "Buy Stop pending order",
        ORDER_TYPE_SELL_STOP: "Sell Stop pending order",
        ORDER_TYPE_BUY_STOP_LIMIT: "Upon reaching the order price, a pending Buy Limit order is placed at the StopLimit price",
        ORDER_TYPE_SELL_STOP_LIMIT: "Upon reaching the order price, a pending Sell Limit order is placed at the StopLimit price",
        ORDER_TYPE_CLOSE_BY: "Order to close a position by an opposite one"
    }

    ORDER_STATE_MAP = {
        ORDER_STATE_STARTED: "Order checked, but not yet accepted by broker",
        ORDER_STATE_PLACED: "Order accepted",
        ORDER_STATE_CANCELED: "Order canceled by client",
        ORDER_STATE_PARTIAL: "Order partially executed",
        ORDER_STATE_FILLED: "Order fully executed",
        ORDER_STATE_REJECTED: "Order rejected",
        ORDER_STATE_EXPIRED: "Order expired",
        ORDER_STATE_REQUEST_ADD: "Order is being registered (placing to the trading system)",
        ORDER_STATE_REQUEST_MODIFY: "Order is being modified (changing its parameters)",
        ORDER_STATE_REQUEST_CANCEL: "Order is being deleted (deleting from the trading system)"
    }

    BUY_ACTIONS = {
        ORDER_TYPE_BUY,
        ORDER_TYPE_BUY_LIMIT,
        ORDER_TYPE_BUY_STOP,
        # ORDER_TYPE_BUY_STOP_LIMIT,
    }

    SELL_ACTIONS = {
        ORDER_TYPE_SELL,
        ORDER_TYPE_SELL_LIMIT,
        ORDER_TYPE_SELL_STOP,
        # ORDER_TYPE_SELL_STOP_LIMIT,
    }

    STRING2TIMEFRAME_MAP = {
        "M1": TIMEFRAME_M1,
        "M2": TIMEFRAME_M2,
        "M3": TIMEFRAME_M3,
        "M4": TIMEFRAME_M4,
        "M5": TIMEFRAME_M5,
        "M6": TIMEFRAME_M6,
        "M10": TIMEFRAME_M10,
        "M12": TIMEFRAME_M12,
        "M15": TIMEFRAME_M15,
        "M20": TIMEFRAME_M20,
        "M30": TIMEFRAME_M30,
        "H1": TIMEFRAME_H1,
        "H2": TIMEFRAME_H2,
        "H3": TIMEFRAME_H3,
        "H4": TIMEFRAME_H4,
        "H6": TIMEFRAME_H6,
        "H8": TIMEFRAME_H8,
        "H12": TIMEFRAME_H12,
        "Daily": TIMEFRAME_D1,
        "Weekly": TIMEFRAME_W1,
        "Monthly": TIMEFRAME_MN1,
    }

    # Reverse map
    TIMEFRAME2STRING_MAP = {v: k for k, v in STRING2TIMEFRAME_MAP.items()}