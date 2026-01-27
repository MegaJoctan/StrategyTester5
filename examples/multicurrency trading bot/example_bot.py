import MetaTrader5 as mt5
from strategytester5.tester import StrategyTester
from strategytester5.trade_classes.Trade import CTrade
import json
import os
import sys

if not mt5.initialize(): # Initialize MetaTrader5 instance
    print(f"Failed to Initialize MetaTrader5. Error = {mt5.last_error()}")
    mt5.shutdown()
    quit()
    
# Get path to the folder where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, "tester.json"), 'r', encoding='utf-8') as file: # reading a JSON file
        # Deserialize the file data into a Python object
        tester_configs = json.load(file)
except Exception as e:
    raise RuntimeError(e)

tester = StrategyTester(tester_config=tester_configs["tester"], mt5_instance=mt5) # very important

# ---------------------- inputs ----------------------------

symbol = "EURUSD"
timeframe = "PERIOD_H1"
magic_number = 10012026
slippage = 100
sl = 1000
tp = 100

# ---------------------------------------------------------

m_trade = CTrade(simulator=tester, magic_number=magic_number, filling_type_symbol=symbol, deviation_points=slippage)
symbol_info = tester.symbol_info(symbol=symbol)

def pos_exists(magic: int, type: int) -> bool:

    for position in tester.positions_get():
        if position.type == type and position.magic == magic:
            return True
    
    return False

def on_tick():
    
    tick_info = tester.symbol_info_tick(symbol=symbol)
    
    ask = tick_info.ask
    bid = tick_info.bid
    
    pts = symbol_info.point
    
    if not pos_exists(magic=magic_number, type=mt5.POSITION_TYPE_BUY): # If a position of such kind doesn't exist
        m_trade.buy(volume=0.1, symbol=symbol, price=ask, sl=ask-sl*pts, tp=ask+tp*pts, comment="Tester buy") # we open a buy position
    
    if not pos_exists(magic=magic_number, type=mt5.POSITION_TYPE_SELL): # If a position of such kind doesn't exist
        m_trade.sell(volume=0.1, symbol=symbol, price=bid, sl=bid+sl*pts, tp=bid-tp*pts, comment="Tester sell") # we open a sell position
    

tester.OnTick(ontick_func=on_tick) # very important!