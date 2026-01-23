from datetime import datetime, timezone, timedelta
import polars as pl
from strategytester5 import *
import os

def bars_to_polars(bars):
    
    return pl.DataFrame({
        "time": bars["time"],
        "open": bars["open"],
        "high": bars["high"],
        "low": bars["low"],
        "close": bars["close"],
        "tick_volume": bars["tick_volume"],
        "spread": bars["spread"],
        "real_volume": bars["real_volume"],
    })
        

def fetch_historical_bars(
                        which_mt5: MetaTrader5,
                        symbol: str,
                        timeframe: int,
                        start_datetime: datetime,
                        end_datetime: datetime) -> pl.DataFrame:
    """
        if not ensure_symbol(symbol=symbol):
            
            if LOGGER is None:
                print(f"Symbol {symbol} not available")
            else:
                LOGGER.warning(f"Symbol {symbol} not available")
            return pl.DataFrame()
    """
    
    start_datetime = ensure_utc(start_datetime)
    end_datetime   = ensure_utc(end_datetime)

    current = start_datetime.replace(day=1, hour=0, minute=0, second=0)

    dfs: list[pl.DataFrame] = []

    tf_name = TIMEFRAMES_MAP_REVERSE[timeframe]

    while True:
        month_start, month_end = month_bounds(current)

        if (
            month_start.year == end_datetime.year and
            month_start.month == end_datetime.month
        ):
            month_end = end_datetime

        if month_start > end_datetime:
            break

        if LOGGER is None:
            print(f"Processing bars for {symbol} ({tf_name}): {month_start:%Y-%m-%d} -> {month_end:%Y-%m-%d}")
        else:
            LOGGER.info(f"Processing bars for {symbol} ({tf_name}): {month_start:%Y-%m-%d} -> {month_end:%Y-%m-%d}")
        

        rates = which_mt5.copy_rates_range(
            symbol,
            timeframe,
            month_start,
            month_end
        )

        if rates is None:
            
            if LOGGER is None:
                print(f"No bars for {symbol} {tf_name} {month_start:%Y-%m}")
            else:
                LOGGER.warning(f"No bars for {symbol} {tf_name} {month_start:%Y-%m}")
                
            current = (month_start + timedelta(days=32)).replace(day=1)
            continue

        df = bars_to_polars(rates)

        df = df.with_columns(
            pl.from_epoch("time", time_unit="s")
              .dt.replace_time_zone("utc")
              .alias("time")
        )

        df = df.with_columns([
            pl.col("time").dt.year().alias("year"),
            pl.col("time").dt.month().alias("month"),
        ])

        df.write_parquet(
            os.path.join("History","Bars", symbol, tf_name),
            partition_by=["year", "month"],
            mkdir=True
        )
        
        # if is_debug: 
        #     print(df.head(-10))
            
        dfs.append(df)

        current = (month_start + timedelta(days=32)).replace(day=1)

    if not dfs:
        return pl.DataFrame()

    return pl.concat(dfs, how="vertical")

"""
if __name__ == "__main__":
    
    if not mt5.initialize():
        print(f"Failed to Initialize MetaTrader5. Error = {mt5.last_error()}")
        mt5.shutdown()
        quit()
    
    start_date = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end_date = datetime.now(tz=timezone.utc)
    
    fetch_historical_bars("XAUUSD", mt5.TIMEFRAME_M1, start_date, end_date)
    fetch_historical_bars("EURUSD", mt5.TIMEFRAME_H1, start_date, end_date)
    fetch_historical_bars("GBPUSD", mt5.TIMEFRAME_M5, start_date, end_date)
    
    # read polaris dataframe and print the head for both symbols

    symbol = "GBPUSD"
    timeframe = TIMEFRAMES_REV[mt5.TIMEFRAME_M5]
    
    path = os.path.join(BARS_HISTORY_DIR, symbol, timeframe)
    
    lf = pl.scan_parquet(path)

    jan_2024 = (
        lf
        .filter(
            (pl.col("year") == 2024) &
            (pl.col("month") == 1)
        )
        .collect(engine="streaming")
    )

    print("January 2024:\n", jan_2024.head(-10))
    print(
        jan_2024.select([
            pl.col("time").min().alias("time_min"),
            pl.col("time").max().alias("time_max")
        ])
    )

    
    mt5.shutdown()
"""