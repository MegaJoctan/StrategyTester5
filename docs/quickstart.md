---
title: StrategyTester5 Quick Start
description: An easy way to start building and testing Python-based trading robots for the MetaTrader5 platform
keywords: StrategyTester5 quickstart, Python trading robot, MetaTrader5 backtesting, MT5 Python trading bot, algorithmic trading Python MT5
---

# Quick Start | Build & Test your First Trading Robot

Every trading robot project needs atleast two files. The main Python file and a JSON configuration file.

![essential files](images/botandjson.png "bot.py config.json")

These files can be named anything but it a good practice to use bot for the main trading robot and tester.json for the configuration file as it contains the so-called *tester configurations*.

## Tester Configurations

```json title="tester.json"
{
    "tester": {
        "bot_name": "Simple Trading Robot",
        "symbols": ["USDJPY"],
        "timeframe": "H1",
        "start_date": "01.01.2026 00:00",
        "end_date": "27.03.2026 00:00",
        "modelling" : "1 minute OHLC",
        "deposit": 1000,
        "leverage": "1:100"
    }
}
```

These configurations resemble those needed to backtest a trading robot in MetaTrader5 GUI.
![mt5config](images/mt5tester%20config.png)


## Nescessary Imports

Inside the main Python script (trading robot), you start by importing all necessary modules

```python title="bot.py"
import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)  # insert(0) so it wins over other paths

from strategytester5.tester import StrategyTester
import MetaTrader5 as mt5
from strategytester5.trade_classes.Trade import CTrade
import json
```

## Importing JSON configurations (Optional)

Instead of importing tester configurations from a JSON file, you could have a dictionary with the right keys and values.

In this case, we have configurations from a JSON file in the same directory (folder) as the main script *bot.py* so, let's import those.

```python title="bot.py"
# Get path to the folder where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, "tester.json"), 'r', encoding='utf-8') as file: # reading a JSON file
        # Deserialize the file data into a Python object
        tester_configs = json.load(file)
except Exception as e:
    raise RuntimeError(e)
```

## Initializing MetaTrader5

Despite the strategy tester simulating the MetaTrader5 terminal, it still relies of the platform for crucial information from instruments (symbols) and a broker's account.

So, we have to initialize the MetaTrader5 terminal using it's native API as we'll need it later.

```python title="bot.py"
if not mt5.initialize():
    raise RuntimeError("Failed to initialize mt5.")
```

## StrategyTester Initialization

```python

tester = StrategyTester(tester_config=tester_configs["tester"], mt5_instance=mt5, logging_level=logging.DEBUG)
```

We also initialize the class CTrade (a submodule for easy and effective ways to manage trades)

```python title="bot.py"
# ---------------------- variables/optional ----------------------------

symbol = "USDJPY" # it should be one among the symbols in symbols list from tester.json (config file/dictionary)
timeframe = "PERIOD_H1"
magic_number = 10012026
slippage = 100
sl = 700
tp = 500

# ---------------------------------------------------------

m_trade = CTrade(terminal=tester.simulated_mt5, magic_number=magic_number, filling_type_symbol=symbol, deviation_points=slippage, logger=tester.logger)
```

## Trading Strategy

```python title="bot.py"
def pos_exists(magic: int, type: int) -> bool:
    
    """Check if position exists"""

    positions_found = tester.simulated_mt5.positions_get()
    for position in positions_found:
        if position.type == type and position.magic == magic:
            return True

    return False

symbol_info = tester.simulated_mt5.symbol_info(symbol=symbol)

def on_tick():
    
    """The main function that gets called upon the receival of new tick(s)"""

    tick_info = tester.simulated_mt5.symbol_info_tick(symbol=symbol)
    if tick_info is None:
        return

    ask = tick_info.ask
    bid = tick_info.bid
    
    pts = symbol_info.point

    if not pos_exists(magic=magic_number, type=mt5.POSITION_TYPE_BUY):  # If a position of such kind doesn't exist
        m_trade.buy(volume=0.01, symbol=symbol, price=ask, sl=ask - sl * pts, tp=ask + tp * pts, comment="Tester buy")  # we open a buy position

    if not pos_exists(magic=magic_number, type=mt5.POSITION_TYPE_SELL):  # If a position of such kind doesn't exist
        m_trade.sell(volume=0.01, symbol=symbol, price=bid, sl=bid + sl * pts, tp=bid - tp * pts, comment="Tester sell")  # we open a sell position
```

## Running the Backtest

All it takes is one function call.

```python
tester.run(on_tick_function=on_tick) # very important!
```