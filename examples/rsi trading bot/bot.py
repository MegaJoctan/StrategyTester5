import logging
from strategytester5.MetaTrader5.api import VirtualMetaTrader5
from strategytester5.tester import run_backtesting
from strategytester5.trade_classes.Trade import CTrade
import MetaTrader5 as parent_mt5
import pandas as pd
from ta.momentum import rsi
import sys

if not parent_mt5.initialize():
    raise RuntimeError(f"Failed to initialize mt5. Error = {parent_mt5.last_error()}")

script_argument = sys.argv[1:]
if "--backtesting" in script_argument:
    mt5 = VirtualMetaTrader5(parent_mt5=parent_mt5) # Assign parent MetaTrader5 to the virtual MetaTrader5 class object
else:
    mt5 = parent_mt5

# ---------------------- inputs/global variables ----------------------------

symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_H1 # This should be an integer so you should convert timeframe in string into integer

# ---------------------------------------------------------

MAGIC_NUMBER = 1001
m_trade = CTrade(terminal=mt5, symbol=symbol, magic_number=MAGIC_NUMBER, deviation_points=100)

def pos_exists(magic: int, pos_type: int) -> bool:
    """Check if position exists"""
    positions_found = mt5.positions_get()
    for position in positions_found:
        if position.type == pos_type and position.magic == magic:
            return True

    return False

def close_pos_by_type(magic: int, pos_type: int):
    """Close positions by type"""
    positions_found = mt5.positions_get()
    for position in positions_found:
        if position.type == pos_type and position.magic == magic:
            m_trade.position_close(position.ticket)


def main():

    indicator_window = 14
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, indicator_window)

    if rates is None or len(rates) < indicator_window: # if no information was found, or less than expected rates were returned
        return # prevent further calculations

    rates_df = pd.DataFrame(data=rates)
    rsi_value = rsi(close=rates_df["close"], window=indicator_window).iloc[-1]

    # rsi strategy

    rsi_oversold = 30.0
    rsi_overbought = 70.0

    symbol_info = mt5.symbol_info(symbol)
    lot_size = symbol_info.volume_min


    if rsi_value < rsi_oversold: # long signal
        if not pos_exists(magic=MAGIC_NUMBER, pos_type=mt5.POSITION_TYPE_BUY):
            m_trade.buy(volume=lot_size, price=symbol_info.ask)

        close_pos_by_type(magic=MAGIC_NUMBER, pos_type=mt5.POSITION_TYPE_SELL)

    if rsi_value > rsi_overbought: # short signal
        if not pos_exists(magic=MAGIC_NUMBER, pos_type=mt5.POSITION_TYPE_SELL):
            m_trade.sell(volume=lot_size, price=symbol_info.bid)

        close_pos_by_type(magic=MAGIC_NUMBER, pos_type=mt5.POSITION_TYPE_BUY)


tester_config = {
        "bot_name": "RSI Strategy Bot",
        "symbols": ["EURUSD"],
        "timeframe": "H1",
        "start_date": "01.01.2026",
        "end_date": "01.06.2026",
        "modelling" : "open price only",
        "deposit": 1000,
        "leverage": "1:100"
}

if "--backtesting" in script_argument:

    stats = run_backtesting(
        main_function=main,
        tester_config=tester_config,
        virtual_mt5=mt5,
        logging_level=logging.DEBUG
    )

else:
    # run the script on the market (realtime)
    while True:
        main()