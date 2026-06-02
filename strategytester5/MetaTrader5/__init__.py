from collections import namedtuple
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

@dataclass(frozen=True)
class MarginEvent:
    state: str  # "OK" | "MARGIN_CALL" | "STOP_OUT"
    reason: str
    value: float  # compared value: margin_level (%) or margin_free (money)
    call_level: float
    stop_level: float
    mode: int  # 0 percent, 1 money

class TradeRequest(NamedTuple):
    action: int
    magic: int
    order: int
    symbol: str
    volume: float
    price: float
    stoplimit: float
    sl: float
    tp: float
    deviation: int
    type: int
    type_filling: int
    type_time: int
    expiration: int
    comment: str
    position: int
    position_by: int

class OrderSendResult(NamedTuple):
    retcode: int
    deal: int
    order: int
    volume: float
    price: float
    bid: float
    ask: float
    comment: str
    request_id: int
    retcode_external: int
    request: TradeRequest

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
        "margin",  #
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
        "ticket",  # DEAL_TICKET
        "order",  # DEAL_ORDER
        "time",  # DEAL_TIME (seconds)
        "time_msc",  # DEAL_TIME_MSC
        "type",  # DEAL_TYPE
        "entry",  # DEAL_ENTRY
        "magic",  # DEAL_MAGIC
        "position_id",  # DEAL_POSITION_ID
        "reason",  # DEAL_REASON
        "volume",  # DEAL_VOLUME
        "price",  # DEAL_PRICE
        "commission",  # DEAL_COMMISSION
        "swap",  # DEAL_SWAP
        "profit",  # DEAL_PROFIT
        "fee",  # DEAL_FEE
        "symbol",  # DEAL_SYMBOL
        "comment",  # DEAL_COMMENT
        "external_id",  # DEAL_EXTERNAL_ID
        "balance",  # Account balance
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


@dataclass
class SimulatedAccountInfo:
    login: int
    trade_mode: int
    leverage: int
    limit_orders: int
    margin_so_mode: int
    trade_allowed: bool
    trade_expert: bool
    margin_mode: int
    currency_digits: int
    fifo_close: bool
    balance: float
    credit: float
    profit: float
    equity: float
    margin: float
    margin_free: float
    margin_level: int
    margin_so_call: int
    margin_so_so: int
    margin_initial: int
    margin_maintenance: int
    assets: int
    liabilities: int
    commission_blocked: int
    name: str
    server: str
    currency: str
    company: str

TerminalInfo = namedtuple(
    "TerminalInfo",
    [
        "community_account",
        "community_connection",
        "connected",
        "dlls_allowed",
        "trade_allowed",
        "tradeapi_disabled",
        "email_enabled",
        "ftp_enabled",
        "notifications_enabled",
        "mqid",
        "build",
        "maxbars",
        "codepage",
        "ping_last",
        "community_balance",
        "retransmission",
        "company",
        "name",
        "language",
        "path",
        "data_path",
        "commondata_path",
    ],
)

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

RATES_DTYPE = np.dtype([
    ("time", "i8"),
    ("open", "f8"),
    ("high", "f8"),
    ("low", "f8"),
    ("close", "f8"),
    ("tick_volume", "i8"),
    ("spread", "i4"),
    ("real_volume", "i8"),
])

TICKS_DTYPE = np.dtype([
    ("time", "i8"),
    ("bid", "f8"),
    ("ask", "f8"),
    ("last", "f8"),
    ("volume", "i8"),
    ("time_msc", "i8"),
    ("flags", "i4"),
    ("volume_real", "f8"),
])

MULTI_TICKS_DTYPE = np.dtype([
    ("time", "i8"),
    ("bid", "f8"),
    ("ask", "f8"),
    ("last", "f8"),
    ("volume", "i8"),
    ("time_msc", "i8"),
    ("flags", "i4"),
    ("volume_real", "f8"),
    ("symbol_id", "i4"),
])

MULTI_RATES_DTYPE = np.dtype([
    ("time", "i8"),
    ("open", "f8"),
    ("high", "f8"),
    ("low", "f8"),
    ("close", "f8"),
    ("tick_volume", "i8"),
    ("spread", "i4"),
    ("real_volume", "i8"),
    ("symbol_id", "i4"),
])


def evaluate_margin_state(acct: AccountInfo) -> MarginEvent:
    """
    Implements MT4/MT5 semantics:
      - mode 0: compare margin_level (%) to call/stop levels
      - mode 1: compare margin_free (money) to call/stop levels
    """

    mode = int(acct.margin_so_mode)

    """
    mode_map = {
        ACCOUNT_STOPOUT_MODE_PERCENT: "Percent",
        ACCOUNT_STOPOUT_MODE_MONEY: "Money"
    }

    print("margin mode: ", mode_map.get(mode, "Unknown"))
    """

    call_level = float(acct.margin_so_call or 0.0)
    stop_level = float(acct.margin_so_so or 0.0)

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
            return MarginEvent("STOP_OUT", f"free_margin {value:.2f} <= stop_out {stop_level:.2f}", value,
                               call_level, stop_level, mode)

        if call_level > 0 and value <= call_level:
            return MarginEvent("MARGIN_CALL", f"free_margin {value:.2f} <= margin_call {call_level:.2f}", value,
                               call_level, stop_level, mode)

        return MarginEvent("OK", "margin ok", value, call_level, stop_level, mode)

    # Unknown mode -> don’t stop out
    return MarginEvent("OK", f"unknown margin_so_mode={mode}", float("inf"), call_level, stop_level, mode)

