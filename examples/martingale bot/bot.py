import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)  # insert(0) so it wins over other paths

from strategytester5.tester import StrategyTester, MetaTrader5
from strategytester5.trade_classes.Trade import CTrade
import json
from datetime import datetime
    
# Get path to the folder where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, "tester.json"), 'r', encoding='utf-8') as file: # reading a JSON file
        # Deserialize the file data into a Python object
        configs_json = json.load(file)
except Exception as e:
    raise RuntimeError(e)

tester_configs = configs_json["tester"]
tester = StrategyTester(tester_config=tester_configs) # very important

# ---------------------- inputs ----------------------------

symbol = "EURUSD"
timeframe = "PERIOD_H1"
magic_number = 10012026
slippage = 100
sl = 700
tp = 500

# ---------------------------------------------------------

m_trade = CTrade(simulator=tester, magic_number=magic_number, filling_type_symbol=symbol, deviation_points=slippage)
symbol_info = tester.symbol_info(symbol=symbol)

def pos_exists(magic: int, type: int) -> bool:

    for position in tester.positions_get():
        if position.type == type and position.magic == magic:
            return True
    
    return False

def martingale_lotsize(initial_lot: float, current_time: datetime, multiplier: float=2) -> float:

    end_date = datetime.strptime(tester_configs["start_date"], "%d.%m.%Y %H:%M")
    deals = tester.history_deals_get(date_from=end_date, date_to=current_time)

    if not deals:
        return initial_lot

    last_deal = deals[-1]

    if last_deal.entry == MetaTrader5.DEAL_ENTRY_OUT: # a closed operation
        if last_deal.profit < 0: # if the deal made a loss
            return last_deal.volume * multiplier

    return initial_lot


def on_tick():
    
    tick_info = tester.symbol_info_tick(symbol=symbol)
    
    ask = tick_info.ask
    bid = tick_info.bid
    curr_time = tick_info.time

    pts = symbol_info.point

    lotsize = 0.01
    final_lotsize = martingale_lotsize(initial_lot=lotsize, current_time=curr_time, multiplier=2)

    if not pos_exists(magic=magic_number, type=MetaTrader5.POSITION_TYPE_BUY):  # If a position of such kind doesn't exist
        m_trade.buy(volume=final_lotsize, symbol=symbol, price=ask, sl=ask - sl * pts, tp=ask + tp * pts, comment="Tester buy")  # we open a buy position

    if not pos_exists(magic=magic_number, type=MetaTrader5.POSITION_TYPE_SELL):  # If a position of such kind doesn't exist
        m_trade.sell(volume=final_lotsize, symbol=symbol, price=bid, sl=bid + sl * pts, tp=bid - tp * pts, comment="Tester sell")  # we open a sell position

tester.OnTick(ontick_func=on_tick) # very important!