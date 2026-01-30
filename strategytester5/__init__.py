__version__ = '1.5.0'
__author__  = 'Omega Joctan Msigwa.'

from collections import namedtuple
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from calendar import monthrange

try:
    import MetaTrader5 as _mt5
    MT5_AVAILABLE = True

except ImportError:
    from strategytester5.mt5 import constants as _mt5

    print(
        "MetaTrader5 is not installed.\n"
        "On Windows, install it with: pip install strategytester5[mt5]\n"
        "Falling back to bundled MT5 constants."
    )
    MT5_AVAILABLE = False

MetaTrader5 = _mt5

def no_mt5_runtime_error():
    if not MT5_AVAILABLE:
        raise RuntimeError(
            "MetaTrader5 is not installed.\n"
            "On Windows, install it with: pip install strategytester5[mt5]"
        )

from typing import Any

IS_DEBUG = True

Tick = namedtuple(
    "Tick",
    [
        "time",
        "bid",
        "ask",
        "last",
        "volume",
        "time_msc",
        "flags",
        "volume_real",
    ]
)

def ensure_symbol(mt5_instance: Any, symbol: str) -> bool:
    info = mt5_instance.symbol_info(symbol)
    if info is None:
        print(f"Symbol {symbol} not found")
        return False

    if not info.visible:
        if not mt5_instance.symbol_select(symbol, True):
            print(f"Failed to select symbol {symbol}")
            return False
    return True

def bytestoMB(size_in_bytes):
    """Convert bytes to megabytes."""
    return size_in_bytes / (1024 * 1024)

def PeriodSeconds(period: int) -> int:
    """
    Convert MT5 timeframe to seconds.
    Correctly decodes MetaTrader 5 bit flags.
    """

    # Months (0xC000)
    if (period & 0xC000) == 0xC000:
        value = period & 0x3FFF
        return value * 30 * 24 * 3600

    # Weeks (0x8000)
    if (period & 0x8000) == 0x8000:
        value = period & 0x7FFF
        return value * 7 * 24 * 3600

    # Hours / Days (0x4000)
    if (period & 0x4000) == 0x4000:
        value = period & 0x3FFF
        return value * 3600

    # Minutes
    return period * 60

def ensure_utc(dt: datetime) -> datetime:
    """
    Ensure datetime is timezone-aware and in UTC.
    - Naive datetimes are assumed to be UTC
    - Aware datetimes are converted to UTC
    """
    if dt.tzinfo is None:
        # Naive → assume UTC
        return dt.replace(tzinfo=timezone.utc)

    # Aware → convert to UTC if needed
    return dt.astimezone(timezone.utc)

def month_bounds(dt: datetime):
    
    """Return (month_start, month_end) in UTC."""
    
    year, month = dt.year, dt.month
    start = datetime(year, month, 1, tzinfo=timezone.utc)

    last_day = monthrange(year, month)[1]
    end = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)

    return start, end
def make_tick(
    time: datetime,
    bid: float,
    ask: float,
    last: float = 0.0,
    volume: int = 0,
    time_msc: int = 0,
    flags: int = -1,
    volume_real: float = 0.0,
    ) -> Tick:

    # MT5 semantics
    time  = ensure_utc(time)

    return Tick(
        time=time,
        bid=float(bid),
        ask=float(ask),
        last=float(bid if last==0 else last),
        volume=int(volume),
        time_msc=time_msc,
        flags=int(flags),
        volume_real=int(volume_real),
    )

def make_tick_from_dict(data: dict) -> Tick:
    """
    Convert a dict into a Tick namedtuple.
    Accepts MT5-like, Polars, or JSON tick dictionaries.
    """

    # --- time handling ---
    time = data.get("time")

    if isinstance(time, (int, float)):
        # epoch seconds
        time = datetime.fromtimestamp(time, tz=timezone.utc)

    elif isinstance(time, datetime):
        time = ensure_utc(time)

    else:
        raise ValueError("Tick dictionary must contain a valid 'time' field")

    return make_tick(
        time=time,
        bid=data.get("bid", 0.0),
        ask=data.get("ask", 0.0),
        last=data.get("last", 0.0),
        volume=data.get("volume", 0),
        time_msc=data.get("time_msc", 0),
        flags=data.get("flags", -1),
        volume_real=data.get("volume_real", 0.0),
    )
    
def make_tick_from_tuple(data: tuple) -> Tick:
    """
    Convert a tuple-based tick into a Tick namedtuple.
    Extra fields at the end of the tuple are ignored.
    """

    if len(data) < 8:
        raise ValueError("Tick tuple must contain at least 8 elements")

    (
        time,
        bid,
        ask,
        last,
        volume,
        time_msc,
        flags,
        volume_real,
        *_
    ) = data

    # --- time handling ---
    if isinstance(time, (int, float)):
        time = datetime.fromtimestamp(time, tz=timezone.utc)
    elif isinstance(time, datetime):
        time = ensure_utc(time)
    else:
        raise ValueError("Invalid time field in tick tuple")

    return make_tick(
        time=time,
        bid=bid,
        ask=ask,
        last=last,
        volume=volume,
        time_msc=time_msc,
        flags=flags,
        volume_real=volume_real,
    )
    
TradeOrder = namedtuple(
    "TradeOrder",
    [
        "ticket",
        "time_setup",
        "time_setup_msc",
        "time_done",
        "time_done_msc",
        "time_expiration",
        "type",
        "type_time",
        "type_filling",
        "state",
        "magic",
        "position_id",
        "position_by_id",
        "reason",
        "volume_initial",
        "volume_current",
        "price_open",
        "sl",
        "tp",
        "price_current",
        "price_stoplimit",
        "symbol",
        "comment",
        "external_id",
    ]
)

TradePosition = namedtuple(
    "TradePosition",
    [
        "ticket",
        "time",
        "time_msc",
        "time_update",
        "time_update_msc",
        "type",
        "magic",
        "identifier",
        "reason",
        "volume",
        "price_open",
        "sl",
        "tp",
        "price_current",
        "swap",
        "profit",
        "symbol",
        "comment",
        "external_id",
        
        # ---- additional fields ----
        # "last_swap_time", 
        "margin", #
    ]
)

SymbolInfo = namedtuple(
    "SymbolInfo",
    [
        "custom",
        "chart_mode",
        "select",
        "visible",
        "session_deals",
        "session_buy_orders",
        "session_sell_orders",
        "volume",
        "volumehigh",
        "volumelow",
        "time",
        "digits",
        "spread",
        "spread_float",
        "ticks_bookdepth",
        "trade_calc_mode",
        "trade_mode",
        "start_time",
        "expiration_time",
        "trade_stops_level",
        "trade_freeze_level",
        "trade_exemode",
        "swap_mode",
        "swap_rollover3days",
        "margin_hedged_use_leg",
        "expiration_mode",
        "filling_mode",
        "order_mode",
        "order_gtc_mode",
        "option_mode",
        "option_right",
        "bid",
        "bidhigh",
        "bidlow",
        "ask",
        "askhigh",
        "asklow",
        "last",
        "lasthigh",
        "lastlow",
        "volume_real",
        "volumehigh_real",
        "volumelow_real",
        "option_strike",
        "point",
        "trade_tick_value",
        "trade_tick_value_profit",
        "trade_tick_value_loss",
        "trade_tick_size",
        "trade_contract_size",
        "trade_accrued_interest",
        "trade_face_value",
        "trade_liquidity_rate",
        "volume_min",
        "volume_max",
        "volume_step",
        "volume_limit",
        "swap_long",
        "swap_short",
        "margin_initial",
        "margin_maintenance",
        "session_volume",
        "session_turnover",
        "session_interest",
        "session_buy_orders_volume",
        "session_sell_orders_volume",
        "session_open",
        "session_close",
        "session_aw",
        "session_price_settlement",
        "session_price_limit_min",
        "session_price_limit_max",
        "margin_hedged",
        "price_change",
        "price_volatility",
        "price_theoretical",
        "price_greeks_delta",
        "price_greeks_theta",
        "price_greeks_gamma",
        "price_greeks_vega",
        "price_greeks_rho",
        "price_greeks_omega",
        "price_sensitivity",
        "basis",
        "category",
        "currency_base",
        "currency_profit",
        "currency_margin",
        "bank",
        "description",
        "exchange",
        "formula",
        "isin",
        "name",
        "page",
        "path",
    ]
)

TradeDeal = namedtuple(
    "TradeDeal",
    [
        "ticket",        # DEAL_TICKET
        "order",         # DEAL_ORDER
        "time",          # DEAL_TIME (seconds)
        "time_msc",      # DEAL_TIME_MSC
        "type",          # DEAL_TYPE
        "entry",         # DEAL_ENTRY
        "magic",         # DEAL_MAGIC
        "position_id",   # DEAL_POSITION_ID
        "reason",        # DEAL_REASON
        "volume",        # DEAL_VOLUME
        "price",         # DEAL_PRICE
        "commission",    # DEAL_COMMISSION
        "swap",          # DEAL_SWAP
        "profit",        # DEAL_PROFIT
        "fee",           # DEAL_FEE
        "symbol",        # DEAL_SYMBOL
        "comment",       # DEAL_COMMENT
        "external_id",   # DEAL_EXTERNAL_ID
        "balance",       # Account balance
    ]
)


AccountInfo = namedtuple(
    "AccountInfo",
    [
        "login",
        "trade_mode",
        "leverage",
        "limit_orders",
        "margin_so_mode",
        "trade_allowed",
        "trade_expert",
        "margin_mode",
        "currency_digits",
        "fifo_close",
        "balance",
        "credit",
        "profit",
        "equity",
        "margin",
        "margin_free",
        "margin_level",
        "margin_so_call",
        "margin_so_so",
        "margin_initial",
        "margin_maintenance",
        "assets",
        "liabilities",
        "commission_blocked",
        "name",
        "server",
        "currency",
        "company",
    ]
)

from dataclasses import dataclass

@dataclass(frozen=True)
class MarginEvent:
    state: str              # "OK" | "MARGIN_CALL" | "STOP_OUT"
    reason: str
    value: float            # compared value: margin_level (%) or margin_free (money)
    call_level: float
    stop_level: float
    mode: int               # 0 percent, 1 money

def evaluate_margin_state(acct: AccountInfo) -> MarginEvent:
    """
    Implements MT4/MT5 semantics:
      - mode 0: compare margin_level (%) to call/stop levels
      - mode 1: compare margin_free (money) to call/stop levels
    """

    mode = int(acct.margin_so_mode)

    """
    mode_map = {
        MetaTrader5.ACCOUNT_STOPOUT_MODE_PERCENT: "Percent",
        MetaTrader5.ACCOUNT_STOPOUT_MODE_MONEY: "Money"
    }

    print("margin mode: ", mode_map.get(mode, "Unknown"))
    """

    call_level = float(acct.margin_so_call or 0.0)
    stop_level = float(acct.margin_so_so   or 0.0)

    if mode == 0:
        # Percent mode: margin_level is percentage.
        # Typical formula: equity/margin*100 when margin>0.

        used_margin = float(acct.margin or 0.0)
        equity = float(acct.equity or 0.0)

        value = float("inf") if used_margin <= 0.0 else (equity / used_margin) * 100.0

        if stop_level > 0 and value <= stop_level:
            return MarginEvent("STOP_OUT", f"margin_level {value:.2f}% <= stop_out {stop_level:.2f}%", value,
                               call_level, stop_level, mode)

        if call_level > 0 and value <= call_level:
            return MarginEvent("MARGIN_CALL", f"margin_level {value:.2f}% <= margin_call {call_level:.2f}%", value,
                               call_level, stop_level, mode)

        return MarginEvent("OK", "margin ok", value, call_level, stop_level, mode)

    elif mode == 1:
        # Money mode: compare free margin to absolute thresholds.
        value = float(acct.margin_free) if acct.margin_free is not None else float("inf")

        if stop_level > 0 and value <= stop_level:
            return MarginEvent("STOP_OUT", f"free_margin {value:.2f} <= stop_out {stop_level:.2f}", value, call_level, stop_level, mode)

        if call_level > 0 and value <= call_level:
            return MarginEvent("MARGIN_CALL", f"free_margin {value:.2f} <= margin_call {call_level:.2f}", value, call_level, stop_level, mode)

        return MarginEvent("OK", "margin ok", value, call_level, stop_level, mode)

    # Unknown mode -> don’t stop out
    return MarginEvent("OK", f"unknown margin_so_mode={mode}", float("inf"), call_level, stop_level, mode)

SUPPORTED_TESTER_MODELLING = {
                "every_tick",
                "real_ticks",
                "new_bar",
                "1-minute-ohlc"
                }

REQUIRED_TESTER_CONFIG_KEYS = {
            "bot_name",
            "symbols",
            "timeframe",
            "start_date",
            "end_date",
            "modelling",
            "deposit",
            "leverage",
        }

DEAL_TYPE_MAP = {
    MetaTrader5.DEAL_TYPE_BUY: "BUY",
    MetaTrader5.DEAL_TYPE_SELL: "SELL",
    MetaTrader5.DEAL_TYPE_BALANCE: "BALANCE",
    MetaTrader5.DEAL_TYPE_CREDIT: "CREDIT",
    MetaTrader5.DEAL_TYPE_CHARGE: "CHARGE",
    MetaTrader5.DEAL_TYPE_CORRECTION: "CORRECTION",
    MetaTrader5.DEAL_TYPE_BONUS: "BONUS",
    MetaTrader5.DEAL_TYPE_COMMISSION: "COMMISSION",
    MetaTrader5.DEAL_TYPE_COMMISSION_DAILY: "COMMISSION DAILY",
    MetaTrader5.DEAL_TYPE_COMMISSION_MONTHLY: "COMMISSION MONTHLY",
    MetaTrader5.DEAL_TYPE_COMMISSION_AGENT_DAILY: "AGENT COMMISSION DAILY",
    MetaTrader5.DEAL_TYPE_COMMISSION_AGENT_MONTHLY: "AGENT COMMISSION MONTHLY",
    MetaTrader5.DEAL_TYPE_INTEREST: "INTEREST",
    MetaTrader5.DEAL_TYPE_BUY_CANCELED: "BUY CANCELED",
    MetaTrader5.DEAL_TYPE_SELL_CANCELED: "SELL CANCELED"
}


DEAL_ENTRY_MAP = {
    MetaTrader5.DEAL_ENTRY_IN: "IN",
    MetaTrader5.DEAL_ENTRY_OUT: "OUT",
    MetaTrader5.DEAL_ENTRY_INOUT: "INOUT"
}

ORDER_TYPE_MAP = {
    MetaTrader5.ORDER_TYPE_BUY: "Market Buy order",
    MetaTrader5.ORDER_TYPE_SELL: "Market Sell order",
    MetaTrader5.ORDER_TYPE_BUY_LIMIT: "Buy Limit pending order",
    MetaTrader5.ORDER_TYPE_SELL_LIMIT: "Sell Limit pending order",
    MetaTrader5.ORDER_TYPE_BUY_STOP: "Buy Stop pending order",
    MetaTrader5.ORDER_TYPE_SELL_STOP: "Sell Stop pending order",
    MetaTrader5.ORDER_TYPE_BUY_STOP_LIMIT: "Upon reaching the order price, a pending Buy Limit order is placed at the StopLimit price",
    MetaTrader5.ORDER_TYPE_SELL_STOP_LIMIT: "Upon reaching the order price, a pending Sell Limit order is placed at the StopLimit price",
    MetaTrader5.ORDER_TYPE_CLOSE_BY: "Order to close a position by an opposite one"
}


ORDER_STATE_MAP = {            
    MetaTrader5.ORDER_STATE_STARTED: "Order checked, but not yet accepted by broker",
    MetaTrader5.ORDER_STATE_PLACED: "Order accepted",
    MetaTrader5.ORDER_STATE_CANCELED: "Order canceled by client",
    MetaTrader5.ORDER_STATE_PARTIAL: "Order partially executed",
    MetaTrader5.ORDER_STATE_FILLED: "Order fully executed",
    MetaTrader5.ORDER_STATE_REJECTED: "Order rejected",
    MetaTrader5.ORDER_STATE_EXPIRED: "Order expired",
    MetaTrader5.ORDER_STATE_REQUEST_ADD: "Order is being registered (placing to the trading system)",
    MetaTrader5.ORDER_STATE_REQUEST_MODIFY: "Order is being modified (changing its parameters)",
    MetaTrader5.ORDER_STATE_REQUEST_CANCEL: "Order is being deleted (deleting from the trading system)"
}

        
BUY_ACTIONS = {
    MetaTrader5.ORDER_TYPE_BUY,
    MetaTrader5.ORDER_TYPE_BUY_LIMIT,
    MetaTrader5.ORDER_TYPE_BUY_STOP,
    MetaTrader5.ORDER_TYPE_BUY_STOP_LIMIT,
}

SELL_ACTIONS = {
    MetaTrader5.ORDER_TYPE_SELL,
    MetaTrader5.ORDER_TYPE_SELL_LIMIT,
    MetaTrader5.ORDER_TYPE_SELL_STOP,
    MetaTrader5.ORDER_TYPE_SELL_STOP_LIMIT,
}

STRING2TIMEFRAME_MAP = {
    "M1": MetaTrader5.TIMEFRAME_M1,
    "M2": MetaTrader5.TIMEFRAME_M2,
    "M3": MetaTrader5.TIMEFRAME_M3,
    "M4": MetaTrader5.TIMEFRAME_M4,
    "M5": MetaTrader5.TIMEFRAME_M5,
    "M6": MetaTrader5.TIMEFRAME_M6,
    "M10": MetaTrader5.TIMEFRAME_M10,
    "M12": MetaTrader5.TIMEFRAME_M12,
    "M15": MetaTrader5.TIMEFRAME_M15,
    "M20": MetaTrader5.TIMEFRAME_M20,
    "M30": MetaTrader5.TIMEFRAME_M30,
    "H1": MetaTrader5.TIMEFRAME_H1,
    "H2": MetaTrader5.TIMEFRAME_H2,
    "H3": MetaTrader5.TIMEFRAME_H3,
    "H4": MetaTrader5.TIMEFRAME_H4,
    "H6": MetaTrader5.TIMEFRAME_H6,
    "H8": MetaTrader5.TIMEFRAME_H8,
    "H12": MetaTrader5.TIMEFRAME_H12,
    "D1": MetaTrader5.TIMEFRAME_D1,
    "W1": MetaTrader5.TIMEFRAME_W1,
    "MN1": MetaTrader5.TIMEFRAME_MN1,
}

# Reverse map
TIMEFRAME2STRING_MAP = {v: k for k, v in STRING2TIMEFRAME_MAP.items()}

def log_date_suffix():
    return datetime.now(timezone.utc).strftime("%Y%m%d")

LOG_DATE = log_date_suffix()

def get_logger(task_name: str, logfile: str, level=logging.INFO):
    """
        Returns a logger
    """
    logger_name = f"{task_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # already configured

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | [%(filename)s:%(lineno)s - %(funcName)10s() ] => %(message)s"
    )

    file_handler = RotatingFileHandler(
        logfile,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger

CURVES_PLOT_INTERVAL_MINS = 60
