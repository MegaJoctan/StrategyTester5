import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)  # insert(0) so it wins over other paths

from strategytester5.tester import StrategyTester, MetaTrader5
from strategytester5.trade_classes.Trade import CTrade
import json
import os
import logging

# Get path to the folder where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, "tester.json"), 'r', encoding='utf-8') as file: # reading a JSON file
        # Deserialize the file data into a Python object
        configs_json = json.load(file)
except Exception as e:
    raise RuntimeError(e)

tester_configs = configs_json["tester"]
tester = StrategyTester(tester_config=tester_configs, logging_level=logging.INFO) # very important

# ---------------------- inputs ----------------------------

timeframe = "PERIOD_H1"
magic_number = 10012026
slippage = 100
sl = 1000
tp = 100

symbols = tester_configs["symbols"]
timeframes = [MetaTrader5.TIMEFRAME_M15, MetaTrader5.TIMEFRAME_H1, MetaTrader5.TIMEFRAME_H4, MetaTrader5.TIMEFRAME_D1]

# ---------------------------------------------------------

m_trade_objects = {
    symbol: CTrade(
        simulator=tester,
        magic_number=magic_number,
        filling_type_symbol=symbol,
        deviation_points=slippage
    )
    for symbol in symbols
}

def pos_exists(magic: int, symbol: str, type: int) -> bool:

    for position in tester.positions_get():
        if position.type == type and position.magic == magic and position.symbol == symbol:
            return True

    return False

def on_tick_multicurrency(symbol: str):

    for tf in timeframes:
        rates = tester.copy_rates_from_pos(symbol=symbol, timeframe=tf, start_pos=0, count=5)

        if rates is None:
            continue

        if len(rates) == 0:
            continue

    return

    m_trade = m_trade_objects[symbol]
    tick_info = tester.symbol_info_tick(symbol=symbol)

    if tick_info is None:  # if the process of obtaining ticks wasn't successful
        return

    ask = tick_info.ask
    bid = tick_info.bid

    symbol_info = tester.symbol_info(symbol)
    pts = symbol_info.point
    volume = symbol_info.volume_min

    if not pos_exists(magic=magic_number, symbol=symbol, type=mt5.POSITION_TYPE_BUY):  # If a position of such kind doesn't exist
        m_trade.buy(volume=volume, symbol=symbol, price=ask, sl=ask - sl * pts, tp=ask + tp * pts, comment="Tester buy")  # we open a buy position

    if not pos_exists(magic=magic_number, symbol=symbol, type=mt5.POSITION_TYPE_SELL):  # If a position of such kind doesn't exist
        m_trade.sell(volume=volume, symbol=symbol, price=bid, sl=bid + sl * pts, tp=bid - tp * pts, comment="Tester sell")  # we open a sell position

tester.ParallelOnTick(ontick_func=on_tick_multicurrency) # very important!