import logging

import MetaTrader5 as mt5
from strategytester5.MetaTrader5.api import VirtualMetaTrader5
from strategytester5.tester import run_backtesting

import pandas as pd
from strategytester5.MQL5.functions import PeriodSeconds
from strategytester5.trade_classes.Trade import CTrade
import ta.momentum as momentum_indicators
import ta.volatility as volatility_indicators
from typing import Any
import sys

if not mt5.initialize():
    raise RuntimeError(f"Failed to initialize mt5. Error = {mt5.last_error()}")

virtual_mt5 = VirtualMetaTrader5()

if not virtual_mt5.initialize(parent_mt5=mt5):
    print("Failed to initialize VirtualMetaTrader5")

SYMBOLS = ["EURUSD",
           "GBPUSD",
           "AUDUSD",
           "USDJPY",
           "USDCAD",
           "USDCHF",
        ]

MAGIC_NUMBER = 25012026
SLIPPAGE = 100
TIMEFRAME = mt5.TIMEFRAME_M30

def is_new_bar(current_time_secs: int, tf: int=TIMEFRAME):

    tf_seconds = PeriodSeconds(tf)
    return current_time_secs % tf_seconds == 0 # new bar e.g., at 11:00, 12:000, etc.

def pos_exists(mt5_instance: Any, pos_type: int, symbol: str, magic: int=MAGIC_NUMBER) -> int:

    positions = mt5_instance.positions_get()
    if positions:
        for pos in positions:
            if pos.type == pos_type and pos.symbol == symbol and pos.magic == magic:
                return True

    return False

def trade_one_instrument(
        mt5_instance: Any,
        symbol: str,
        timeframe: int,
        rates_object: dict,
        m_trade: CTrade,
        rsi_period: int=14,
        rsi_overbought: int=70,
        rsi_oversold: int=30,
        atr_period: int = 13,
        sl_atr_multiplier: float=2.5,
        tp_atr_multiplier: float=1.5
    ):

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return

    ticks = mt5_instance.symbol_info_tick(symbol)

    rates_window = max(rsi_period, atr_period) # maximum value between indicator periods
    if is_new_bar(ticks.time):

        rates = mt5_instance.copy_rates_from_pos(symbol, timeframe, 0, rates_window)
        rates_object["df"] = pd.DataFrame(rates)

    rates_df = rates_object["df"]
    if rates_df is None or len(rates_df) < rates_window:
        return

    # open = rates_df["open"]
    high = rates_df["high"]
    low = rates_df["low"]
    close = rates_df["close"]

    rsi_value = momentum_indicators.rsi(close=close, window=rsi_period).iloc[-1]
    atr_value = volatility_indicators.average_true_range(high=high, low=low, close=close, window=atr_period).iloc[-1]

    volume = symbol_info.volume_min

    # Trading strategy

    ask = ticks.ask
    bid = ticks.bid

    if rsi_value > rsi_overbought: # sell signal

        if not pos_exists(mt5_instance=mt5_instance, pos_type=mt5_instance.POSITION_TYPE_SELL, symbol=symbol, magic=MAGIC_NUMBER):
            m_trade.sell(volume=volume,
                         price=bid,
                         sl=bid+atr_value*sl_atr_multiplier,
                         tp=bid-atr_value*tp_atr_multiplier)

    elif rsi_value < rsi_oversold: # buy signal

        if not pos_exists(mt5_instance=mt5_instance, pos_type=mt5_instance.POSITION_TYPE_BUY, symbol=symbol, magic=MAGIC_NUMBER):
            m_trade.buy(volume=volume,
                        price=ask,
                        sl=ask-atr_value*sl_atr_multiplier,
                        tp=ask+atr_value*tp_atr_multiplier)


# Final execution according to user settings

if not mt5.initialize():
    raise RuntimeError(f"Failed to initialize mt5. Error = {mt5.last_error()}")

script_args = sys.argv[1:] # Check the parent mode (whether a user wants to backtest, do live trading, or optimization)

if "--live-trading" in script_args:

    m_trade_objects = [
        CTrade(
            magic_number=MAGIC_NUMBER,
            symbol=s,
            terminal=mt5,
            deviation_points=SLIPPAGE
        ) for s in SYMBOLS
    ]

    rates_obj = {
        "df": None
    }

    while True:
        for i, symbol in enumerate(SYMBOLS):
            trade_one_instrument(
                mt5_instance=mt5,
                symbol=symbol,
                timeframe=TIMEFRAME,
                rates_object=rates_obj,
                m_trade=m_trade_objects[i]
            ) # Execute a strategy on several instruments

elif "--backtesting" in script_args:

    tester_config = {
        "bot_name": "Multicurrency Trading Bot",
        "symbols": SYMBOLS,
        "timeframe": "H1",
        "start_date": "01.01.2026",
        "end_date": "01.03.2026",
        "modelling": "1 minute OHLC",
        "deposit": 1000,
        "leverage": "1:500",
    }

    tester = StrategyTester(
        tester_config=tester_config,
        mt5_instance=mt5,
        logging_level=logging.INFO
    )

    simulated_mt5 = tester.simulated_mt5
    logger = tester.logger

    m_trade_objects = [
        CTrade(
            magic_number=MAGIC_NUMBER,
            symbol=s,
            terminal=simulated_mt5,
            deviation_points=SLIPPAGE,
            logger=logger
        ) for s in SYMBOLS
    ]

    rates_obj = {
        "df": None
    }

    def main():
        for i, sym in enumerate(SYMBOLS):
            trade_one_instrument(
                mt5_instance=simulated_mt5,
                symbol=sym,
                timeframe=TIMEFRAME,
                rates_object=rates_obj,

                m_trade=m_trade_objects[i]
            ) # Execute a strategy on several instruments


    tester.run(main, dashboard_fps=30)
    deals_df = pd.DataFrame(simulated_mt5.DEALS).to_csv("deals.csv")
    orders_df = pd.DataFrame(simulated_mt5.ORDERS_HISTORY).to_csv("orders.csv")

# elif "---optimization" in script_args:

