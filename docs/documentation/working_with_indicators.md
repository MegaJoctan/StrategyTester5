---
title: Using Trading Indicators in Python with MetaTrader5 (StrategyTester5 Guide)
description: Learn how to calculate and use trading indicators like SMA in Python with MetaTrader5 and StrategyTester5. Includes examples using pandas and the ta library for backtesting strategies.
keywords: trading indicators python MT5, MetaTrader5 indicators python, SMA indicator python example, StrategyTester5 indicators guide, technical analysis python ta library, algorithmic trading indicators
---

Many trading strategies rely on the so-called *Trading indicators* for example the [RSI trading strategy](../documentation/rsi_strategy.md). 

To calculate an indicator you can implement your custom calculations for it and use it for your trading strategies or you could use any of the available modules for the task, one of them being the [Technical Analysis framework](https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html)

Start by importing the necessary modules.

Pandas for data handling:

```py
import pandas as pd
```

The Technical Analysis (TA) framework:
```py
from ta.trend import sma_indicator
```
In this example we will deploy a simple moving average indicator, so rather than importing the entire `ta` library, we import a simple moving average indicator from the "trend" submodule.

!!! warning "Important"

    It is a good practice to assign the simulated MetaTrader5 instance which has similar methods to the native Python API into a separate (easy to remeber) variable. This variable should give us everything we need from the simulated MetaTrader5 instance.

    ```py
    tester = StrategyTester(tester_config=tester_config, mt5_instance=mt5, logging_level=logging.DEBUG)
    sim_mt5 = tester.simulated_mt5 # extract the simulated metatrader5 from the StrategyTester object and assign it to a simple variable
    logger = tester.logger # obtain the logger
    ```

Inside your main function, you want to:

## Copy rates from the simulator using a simulated MetaTrader5 

```py

def on_tick():

    indicator_window = 20
    rates = sim_mt5.copy_rates_from_pos(symbol=symbol, timeframe=timeframe, start_pos=0, count=indicator_window)

    if rates is None or len(rates) < indicator_window: # if no information was found, or less than expected rates were returned
        return # prevent further calculations
```

!!! Tip "Tip"

    When copying rates errors and zero-length data can be returned sometimes due to several reasons such as:

    During the begining of a test, where the simulator is yet to see much data.
    
    For example: Strategy tester's starting date is 01.01.2020, during the simulation (in the first hour and early minutes/seconds) we could request 100 bars in the past (before that time), this would cause all methods that returns past data like "copy_rates" methods to either return `None`, or less data than anticipated.

    To prevent further issues, you should always check if enough data is returned before calculating your indicators.
    ```python
    if rates is None or len(rates) < indicator_window: # if no information was found, or less than expected rates were returned
        return # prevent further calculations
    ```

    **If insufficient data is returned, the function should skip indicator calculation until enough data is obtained.**

## Indicator Calculation

```py
def on_tick():

    indicator_window = 20
    rates = sim_mt5.copy_rates_from_pos(symbol=symbol, timeframe=timeframe, start_pos=0, count=indicator_window)

    if rates is None or len(rates) < indicator_window: # if no information was found, or less than expected rates were returned
        return # prevent further calculations

    rates_df = pd.DataFrame(data=rates)

    sma_indicator_values = sma_indicator(rates_df["close"], window=indicator_window)

    logger.info(f"Current sma({indicator_window}) : {sma_indicator_values.iloc[-1]}") # print the last calculated value in the array
```

Example outputs:

```bash
StrategyTester Progress:   0%|          | 0/1441 [00:00<?, ?bar/s]2026-01-02 00:00:00 | INFO     | Simple Trading Robot | [bot.py:45 -    on_tick() ] => Current sma(20) : 156.8057
2026-01-02 01:00:00 | INFO     | Simple Trading Robot | [bot.py:45 -    on_tick() ] => Current sma(20) : 156.81225
2026-01-02 02:00:00 | INFO     | Simple Trading Robot | [bot.py:45 -    on_tick() ] => Current sma(20) : 156.8277
2026-01-02 03:00:00 | INFO     | Simple Trading Robot | [bot.py:45 -    on_tick() ] => Current sma(20) : 156.83785
2026-01-02 04:00:00 | INFO     | Simple Trading Robot | [bot.py:45 -    on_tick() ] => Current sma(20) : 156.83785
2026-01-05 00:00:00 | INFO     | Simple Trading Robot | [bot.py:45 -    on_tick() ] => Current sma(20) : 156.833
2026-01-05 01:00:00 | INFO     | Simple Trading Robot | [bot.py:45 -    on_tick() ] => Current sma(20) : 156.7946
2026-01-05 02:00:00 | INFO     | Simple Trading Robot | [bot.py:45 -    on_tick() ] => Current sma(20) : 156.754
2026-01-05 03:00:00 | INFO     | Simple Trading Robot | [bot.py:45 -    on_tick() ] => Current sma(20) : 156.7232
2026-01-05 04:00:00 | INFO     | Simple Trading Robot | [bot.py:45 -    on_tick() ] => Current sma(20) : 156.69075
StrategyTester Progress:   2%|▏         | 33/1441 [00:00<00:04, 326.34bar/s]2026-01-05 05:00:00 | INFO     | Simple Trading Robot | [bot.py:45 -    on_tick() ] => Current sma(20) : 156.6599
```
