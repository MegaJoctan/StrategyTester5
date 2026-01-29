from strategytester5.hist.manager import HistoryManager
from strategytester5.validators.tester_configs import TesterConfigValidators
from strategytester5 import get_logger, logging, STRING2TIMEFRAME_MAP

history_dir = "Test History"
    
if __name__ == "__main__":

    tester_config: dict = {
        "bot_name": "test EA",
        "deposit": 1000,
        "leverage": "1:100",
        "timeframe": "H1",
        "start_date": "01.01.2025 00:00",
        "end_date": "31.12.2025 00:00",
        "modelling": "1-minute-ohlc",
        # "modelling": "new_bar",
        "symbols": ["EURUSD", "GBPUSD", "USDCAD"]
    }

    tester_config = TesterConfigValidators.parse_tester_configs(tester_config)
        
    start_dt = tester_config["start_date"]
    end_dt   = tester_config["end_date"]
    symbols = tester_config["symbols"]
    timeframe = tester_config["timeframe"]
    modelling = tester_config["modelling"]


    import MetaTrader5 as mt5
    

    if not mt5.initialize():
        print("Failed to initialize MetaTrader5, Error = ", mt5.last_error())
        quit()

    import strategytester5.hist.bars as bars
    import strategytester5.hist.ticks as ticks

    logger = get_logger("test",
                         logfile= f"test.log",
                         level=logging.DEBUG)

    history_dir = r'D:\StrategyTester5\examples\multicurrency trading bot\History'
    ticks_df = ticks.get_ticks_from_history(
        symbol=symbols[0],
        start_datetime=start_dt,
        end_datetime=end_dt,
        POLARS_COLLECT_ENGINE="auto",
        logger=logger,
        hist_dir=history_dir
    )

    print(ticks_df.head(-10))

    bars_df = bars.get_bars_from_history(
        symbol=symbols[0],
        timeframe=timeframe,
        start_datetime=start_dt,
        end_datetime=end_dt,
        POLARS_COLLECT_ENGINE="auto",
        logger=logger,
        hist_dir=history_dir
    )

    print(bars_df.head(-10))

    """
    hist_manager = HistoryManager(mt5_instance=mt5,
                                start_dt=start_dt,
                                end_dt=end_dt,
                                symbols=symbols,
                                timeframe=timeframe)

    hist_manager.fetch_history(modelling=modelling)
    hist_manager.synchronize_timeframes()
    """