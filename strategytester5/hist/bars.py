from datetime import datetime, timezone, timedelta
import polars as pl
from strategytester5 import LOGGER, MetaTrader5, ensure_utc, TIMEFRAME2STRING_MAP, month_bounds
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
                        end_datetime: datetime,
                        hist_dir: str="History",
                        return_df: bool = False
                        ) -> pl.DataFrame:
    
    start_datetime = ensure_utc(start_datetime)
    end_datetime   = ensure_utc(end_datetime)

    current = start_datetime.replace(day=1, hour=0, minute=0, second=0)

    dfs: list[pl.DataFrame] = []

    tf_name = TIMEFRAME2STRING_MAP[timeframe]

    while True:
        month_start, month_end = month_bounds(current)

        if (
            month_start.year == end_datetime.year and
            month_start.month == end_datetime.month
        ):
            month_end = end_datetime

        if month_start > end_datetime:
            # if LOGGER is None:
            #     print("start > end")
            # else:
            #     LOGGER.debug("start > end")
            break

        if LOGGER is None:
            print(f"\nProcessing bars for {symbol} ({tf_name}): {month_start:%Y-%m-%d} -> {month_end:%Y-%m-%d}")
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
                print(f"\nNo bars for {symbol} {tf_name} {month_start:%Y-%m}")
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
            os.path.join(hist_dir, "Bars", symbol, tf_name),
            partition_by=["year", "month"],
            mkdir=True
        )
        
        # if IS_DEBUG:
        #    print(df.head(-10))
        
        if return_df:    
            dfs.append(df)

        current = (month_start + timedelta(days=32)).replace(day=1)

    return pl.concat(dfs, how="vertical") if return_df else None
