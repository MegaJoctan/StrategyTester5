import logging
from strategytester5.MetaTrader5.api import VirtualMetaTrader5
from strategytester5.tester import run_backtesting
from strategytester5.trade_classes.Trade import CTrade
import MetaTrader5 as parent_mt5
from datetime import datetime
import sys

if not parent_mt5.initialize():
    raise RuntimeError(f"Failed to initialize mt5. Error = {parent_mt5.last_error()}")

script_argument = sys.argv[1:]
if "--backtesting" in script_argument:
    mt5 = VirtualMetaTrader5(parent_mt5=parent_mt5) # Assign parent MetaTrader5 to the virtual MetaTrader5 class object
else:
    mt5 = parent_mt5

# ---------------------- inputs ----------------------------

symbol = "EURUSD"
timeframe = "PERIOD_H1"
magic_number = 10012026
slippage = 100
sl = 700
tp = 500

# strategytester5 configurations

tester_configs = {
        "bot_name": "martingale-based trading bot",
        "symbols": ["EURUSD"],
        "timeframe": "H1",
        "start_date": "01.01.2026",
        "end_date": "01.06.2026",
        "modelling" : "1 minute ohlc",
        "deposit": 1000,
        "leverage": "1:100"
}
# ---------------------------------------------------------

m_trade = CTrade(terminal=mt5, magic_number=magic_number, symbol=symbol, deviation_points=slippage)
symbol_info = mt5.symbol_info(symbol=symbol)

def pos_exists(magic: int, pos_type: int) -> bool:

    for position in mt5.positions_get():
        if position.type == pos_type and position.magic == magic:
            return True
    
    return False

def martingale_lot_size(initial_lot: float, current_time: datetime, multiplier: float=2) -> float:

    end_date = datetime.strptime(tester_configs["start_date"], "%d.%m.%Y")
    deals = mt5.history_deals_get(date_from=end_date, date_to=current_time)

    if not deals:
        return initial_lot

    last_deal = deals[-1]

    if last_deal.entry == mt5.DEAL_ENTRY_OUT: # a closed operation
        if last_deal.profit < 0: # if the deal made a loss
            return last_deal.volume * multiplier

    return initial_lot


def main():
    
    tick_info = mt5.symbol_info_tick(symbol=symbol)
    
    ask = tick_info.ask
    bid = tick_info.bid
    curr_time = tick_info.time

    pts = symbol_info.point

    lot_size = 0.01
    final_lot_size = martingale_lot_size(initial_lot=lot_size, current_time=curr_time, multiplier=2)

    if not pos_exists(magic=magic_number, pos_type=mt5.POSITION_TYPE_BUY):  # If a position of such kind doesn't exist
        m_trade.buy(volume=final_lot_size, price=ask, sl=ask - sl * pts, tp=ask + tp * pts, comment="Tester buy")  # we open a buy position

    if not pos_exists(magic=magic_number, pos_type=mt5.POSITION_TYPE_SELL):  # If a position of such kind doesn't exist
        m_trade.sell(volume=final_lot_size, price=bid, sl=bid + sl * pts, tp=bid - tp * pts, comment="Tester sell")  # we open a sell position

if "--backtesting" in script_argument:

    stats = run_backtesting(
        main_function=main,
        tester_config=tester_configs,
        virtual_mt5=mt5,
        logging_level=logging.DEBUG
    )

else:
    # run the script on the market (realtime)
    while True:
        main()