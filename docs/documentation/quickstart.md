---
title: StrategyTester5 Quick Start
description: An easy way to start building and testing Python-based trading robots for the MetaTrader5 platform
keywords: StrategyTester5 quickstart, Python trading robot, MetaTrader5 backtesting, MT5 Python trading bot, algorithmic trading Python MT5
---

# Quick Start | Build & Test your First Trading Robot

All you need is to add three (3) Lines of code to your existing Python project.

## 01: Import the VirtualMetaTrader5 class

```Python
from strategytester5.MetaTrader5.api import VirtualMetaTrader5
```

## 02: Assign the parent (MetaTrader5) initialized instance to the VirtulMetaTrader5 object
```Python
if not mt5.initialize():
    raise RuntimeError(f"Failed to initialize mt5. Error = {mt5.last_error()}")

virtual_mt5 = VirtualMetaTrader5(parent_mt5=mt5)
```

## Optional : Choosing the right MT5 instance
Since the VirtualMetaTrader5 object has similar methods and constants to MetaTrader5 API.

To keep all the code in a single file, introduce if statements to check for passed arguments from the script.

!!! Note "Rule of Thumb"
    
    For instance, if a user has passed `--backtesting` when calling the main script the script should use a simulated MetaTrader5 rather than the actual one.

    ```python
    script_argument = sys.argv[1:]
    if "--backtesting" in script_argument:
        mt5 = VirtualMetaTrader5(parent_mt5=parent_mt5) # Assign parent MetaTrader5 to the virtual MetaTrader5 class object
    else:
        mt5 = parent_mt5
    ```

    > Always use the virtual MetaTrader5 object for backtesting and visualization purposes and stick to the original one for live trading. *That's all*

## 03: Backtesting your Python Trading Strategies

To Backtest your trading systems call the function `run_backtesting`.
```python
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
```

## Tester Configurations

The so-called `tester_config` is supposed to be a dictionary with a set of key, and value pairs that resemble MetaTrader5's strategy tester section.

![mt5config](../images/mt5tester%20config.png)

```python title="example tester config(s)"
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
```
[Learn More.](https://strategytester5.com/documentation/#metatrader5-like-strategytester-configurations)

!!! Note "Rule of Thumb"

    Storing tester configuration in a JSON file is the best way forward. It helps adjust the values without changing the original code :)
    