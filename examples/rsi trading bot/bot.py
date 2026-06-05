import logging
from strategytester5.MetaTrader5.api import VirtualMetaTrader5
from strategytester5.tester import run_backtesting
from strategytester5.trade_classes.Trade import CTrade
import MetaTrader5 as mt5
import pandas as pd
from ta.momentum import rsi

if not mt5.initialize():
    raise RuntimeError(f"Failed to initialize mt5. Error = {mt5.last_error()}")

virtual_mt5 = VirtualMetaTrader5(parent_mt5=mt5)

# ---------------------- inputs ----------------------------

symbol = "EURUSD"
timeframe = virtual_mt5.TIMEFRAME_H1 # This should be an integer so you should convert timeframe in string into integer

# ---------------------------------------------------------

MAGIC_NUMBER = 1001
m_trade = CTrade(terminal=virtual_mt5, symbol=symbol, magic_number=MAGIC_NUMBER, deviation_points=100)

def pos_exists(magic: int, pos_type: int) -> bool:
    """Check if position exists"""
    positions_found = virtual_mt5.positions_get()
    for position in positions_found:
        if position.type == pos_type and position.magic == magic:
            return True

    return False

def close_pos_by_type(magic: int, pos_type: int):
    """Close positions by type"""
    positions_found = virtual_mt5.positions_get()
    for position in positions_found:
        if position.type == pos_type and position.magic == magic:
            m_trade.position_close(position.ticket)

def on_tick():

    indicator_window = 14
    rates = virtual_mt5.copy_rates_from_pos(symbol=symbol, timeframe=timeframe, start_pos=0, count=indicator_window)

    if rates is None or len(rates) < indicator_window: # if no information was found, or less than expected rates were returned
        return # prevent further calculations

    rates_df = pd.DataFrame(data=rates)
    rsi_value = rsi(close=rates_df["close"], window=indicator_window).iloc[-1]

    # rsi strategy

    rsi_oversold = 30.0
    rsi_overbought = 70.0

    symbol_info = virtual_mt5.symbol_info(symbol=symbol)
    lot_size = symbol_info.volume_min


    if rsi_value < rsi_oversold: # long signal
        if not pos_exists(magic=MAGIC_NUMBER, pos_type=virtual_mt5.POSITION_TYPE_BUY):
            m_trade.buy(volume=lot_size, price=symbol_info.ask)

        close_pos_by_type(magic=MAGIC_NUMBER, pos_type=virtual_mt5.POSITION_TYPE_SELL)

    if rsi_value > rsi_overbought: # short signal
        if not pos_exists(magic=MAGIC_NUMBER, pos_type=virtual_mt5.POSITION_TYPE_SELL):
            m_trade.sell(volume=lot_size, price=symbol_info.bid)

        close_pos_by_type(magic=MAGIC_NUMBER, pos_type=virtual_mt5.POSITION_TYPE_BUY)


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


total = 10
for i in range(1, total):
    stats = run_backtesting(
        main_function=on_tick,
        tester_config=tester_config,
        virtual_mt5=virtual_mt5,
        logging_level=logging.DEBUG,
        # is_optimization_mode=True
    )

