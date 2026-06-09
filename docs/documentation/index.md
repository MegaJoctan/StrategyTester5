---
title: StrategyTester5 Usage Guide – Rules, Configuration & MetaTrader5 Integration
description: Learn how to use StrategyTester5 with MetaTrader5 in Python. Covers setup rules, configuration options, simulated MT5 API usage, and best practices for backtesting trading strategies.
keywords: StrategyTester5 guide, MetaTrader5 Python backtesting, MT5 strategy tester rules, trading bot configuration python, algorithmic trading MT5 setup
---

## How StrategyTester5 Works (TL;DR)

![how-it-works](../images/how-it-works.png)

### VirtualMetaTrader5
This is a virtual (simulated) MetaTrader5 API, it has similar methods and constants as MetaTrader5. This is the simulation engine.

### run_backtesting
Calls the main (strategy) execution function from the user and calls it repeatedly on historical data.

## MetaTrader5-Like StrategyTester Configurations

The StrategyTester class expects tester configurations in the form of a Python dictionary:

As it stands currently, supported keys in a configuration dictionary include:

| Key        | Description                                                                                                                                                                                                                   |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| bot_name   | The name of a trading robot you are working on, this name will be used for logging and in naming backtesting reports                                                                                                          |
| symbols    | A list of instruments that are expected to be used in the project. **No surprises allowed — **all symbols must be defined here before deployment**                                                                            |
| timeframe  | The main timeframe you want to use. This doesn't affect trading outcome, but it is mostly used for fetching bars and tells the simulator how often to record balance, equity, and other crucial information during simulation |
| start_date | The starting date to start backtesting                                                                                                                                                                                        |
| end_date   | The ending date for backtesting                                                                                                                                                                                               |
| modelling  | The type of modelling to use (*Case insensitive*). Suppored: `1 minute OHLC`, `Open price only`, and `Every tick based on real ticks`                                                                                         |
| deposit    | Initial account balance for the backtesting                                                                                                                                                                                   |
| leverage   | [Leverage](https://www.ig.com/en/risk-management/what-is-leverage) to use for the simulated account                                                                                                                           |
| visual_mode| ([For the professional version](https://omegajoctan.gumroad.com/l/strategytester5-professional)). Enables strategy visualization in MetaTrader5.                                                                              | 

Example **config.json**

```python
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

## All Trading Logic Should be Organized or Traced from a Single Function

Similarly to the OnTick function in MQL5 programming language, which calls all functions and lines of code that makes up a trading strategy.

We recommend doing the same in your Python script. This final "strategy" function should be passed to the method `run_backtest` which runs the it repetetively depending on [modelling type](https://www.google.com/search?q=mql5+modelling+types+strategy+testing) selected in tester configs.

```python

stats = run_backtesting(
    main_function=main,
    tester_config=tester_config,
    virtual_mt5=mt5,
    logging_level=logging.DEBUG
)
```

Working examples are found [here](https://github.com/MegaJoctan/StrategyTester5/tree/main/examples).