from __future__ import annotations
from typing import Any, Literal, Optional, Dict

import pandas as pd
import polars as pl

import threading
import time

from MetaTrader5 import AccountInfo
from flask import Flask, render_template
from flask_socketio import SocketIO
import webbrowser

from strategytester5.MetaTrader5 import evaluate_margin_state, TradeDeal, Tick
from strategytester5.MetaTrader5.api import VirtualMetaTrader5
from strategytester5.utils import TesterConfigValidators
from strategytester5.MQL5.functions import PeriodSeconds
from . import *
from datetime import datetime
import os
import numpy as np
from . import stats
import logging
from tqdm import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import MetaTrader5
from pathlib import Path
from .stats import TesterStats
import json


class DashboardState:
    live_data: dict = {
        "bot_name": "",
        "time": np.nan,
        "balance": np.nan,
        "equity": np.nan,
        "free_margin": np.nan,
        "trades": []
    }

    tester_stats: dict = {}
    entries_pl_plot: str = ""
    holding_plot: dict = {}
    holding_stats: dict = {}
    simulation_running: bool = True


app = Flask(
    __name__,
    template_folder="../strategytester5/templates",
    static_folder="../strategytester5/static"
)

socketio = SocketIO(
    app,
    # ping_timeout=300,
    cors_allowed_origins="*"
)


# ------------- FLASK ROUTES ---------------

@app.route("/")
def index():
    return render_template("dashboard.html")


class StrategyTester:
    """
        The "engine" that drives the entire backtesting process, simulating the MetaTrader5 environment and allowing you
        to test your trading strategies against historical data. The main method is `run()`,
        which takes a callback function that contains your strategy logic and executes it on each tick or bar,
         depending on the **modelling type** selected in `tester_config`.

        > Similar to the MetaTrader5 strategy tester
    """

    def __init__(self,
                 tester_config: dict,
                 virtual_mt5: VirtualMetaTrader5,
                 logging_level: int = logging.WARNING,
                 logs_dir: Optional[str] = "Logs",
                 trading_history_dir: Optional[str] = "TradesHistory",
                 polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto"):

        """Instantiates the StrategyTester class with the given configuration, sets up the simulated MetaTrader5 environment, and prepares for running the backtest.

        Args:
            tester_config (dict): Dictionary of tester configuration values.
            virtual_mt5 (MetaTrader5): Virtual (simulated) MetaTrader5 instance.
            logging_level: Minimum severity of messages to record. Uses standard `logging` levels (e.g., logging.DEBUG, INFO, WARNING, ERROR, CRITICAL). Messages below this level are ignored.
            logs_dir (str): Directory for log files.
            trading_history_dir (str | optional) A directory to keep trading history.

            polars_collect_engine (str): Engine used by Polars when collecting historical data in functions for obtaining ticks — copy_ticks*, and bars information/rates (copy_rates*). Supported values are:
                - ``"auto"`` (default): Use Polars’ standard in-memory engine and
                    respect the ``POLARS_ENGINE_AFFINITY`` environment variable if set.
                - ``"in-memory"``: Explicitly use the default in-memory engine,
                    optimized with multi-threading and SIMD over Arrow data.
                - ``"streaming"``: Process queries in batches, enabling
                    larger-than-RAM datasets.
                - ``"gpu"``: Use NVIDIA GPUs via RAPIDS cuDF for accelerated execution.
                    Requires installing Polars with GPU support, e.g.:
                    ``pip install polars[gpu] --extra-index-url=https://pypi.nvidia.com``.
        Raises:
            RuntimeError: If required MT5 account info cannot be obtained.
        """

        # ---------------- validate all configs from a dictionary -----------------

        self.tester_config = TesterConfigValidators.parse_tester_configs(tester_config)
        self.ea_name = self.tester_config["bot_name"]

        self.logs_dir = logs_dir
        self.logging_level = logging_level

        os.makedirs(logs_dir, exist_ok=True)

        self.logger = get_logger(task_name=self.ea_name,
                                 logfile=os.path.join(logs_dir, f"{LOG_DATE}.log"),
                                 level=logging_level,
                                 time_provider=self._get_sim_time)

        self.polars_collect_engine = polars_collect_engine
        self.trading_history_dir = trading_history_dir

        self.positions_unrealized_pl = 0
        self.positions_total_margin = 0

        self.virtual_mt5 = virtual_mt5

        vlogger = self.virtual_mt5.logger
        self.virtual_mt5.logger = self.logger if vlogger is None else vlogger  # use the same logger

        self.simulated_mt5 = virtual_mt5  # simulated MetaTrader5 instance

        # -------------------- tester reports ----------------------------

        self.tester_curves = {
            "time": np.array([]),
            "balance": np.array([]),
            "equity": np.array([]),
            "margin_level": np.array([])
        }

        self.TESTER_IDX = 0
        self.CURVES_IDX = 0
        self.IS_MARGIN_STOPOUT = False
        self.IS_MARGIN_CALL = False
        self.IS_OPTIMIZATION_MODE = False
        self.IS_OPTIMIZATION_FIRST_RUN = True  # the first pass of the optimization loop
        self.history_dataframe = None  # polars dataframe query object to be used for the simulation loop
        self.history_manager = virtual_mt5.history_manager  # Get history manager object from Virtual MT5

        # self._engine_lock = threading.RLock()   # re-entrant lock (safe if functions call other locked functions)

        # ---------------------- Dashboard ------------------------------

        self.DASHBOARD_STATE = DashboardState()
        self.last_tick_time: int = 0
        self._last_dashboard_update = 0.0

    def _tester_init(self, is_optimization_mode: bool) -> bool:
        """
        Initializes the StrategyTester class and the simulated MetaTrader5 instance.

        """

        self.virtual_mt5.IS_OPTIMIZATION_MODE = is_optimization_mode
        self.virtual_mt5.assign_logger(self.logger)  # use the same logger for virtual MetaTrader5 as the tester object

        # reset values

        self.TESTER_IDX = 0  # initialize tester curves and the index for incrementing
        self.CURVES_IDX = 0
        self.IS_MARGIN_STOPOUT = False
        self.IS_MARGIN_CALL = False
        self.IS_OPTIMIZATION_MODE = is_optimization_mode

        # -------------------- initialize the Loggers ----------------------------

        start_dt = self.tester_config.get("start_date", 0)
        start_dt_ts = start_dt.timestamp() if isinstance(start_dt, datetime) else start_dt

        self.simulated_mt5._current_time = start_dt_ts
        self.simulated_mt5._current_time_msc = start_dt_ts * 1000

        deposit = self.tester_config["deposit"]

        sim_ac = self.simulated_mt5.ACCOUNT._replace(
            # ---- identity / broker-controlled ----
            login=11223344,
            trade_mode=self.simulated_mt5.ACCOUNT.trade_mode,
            leverage=int(self.tester_config["leverage"]),

            # ---- simulator-controlled financials ----
            balance=deposit,  # simulator starting balance
            credit=0,
            profit=0.0,
            equity=deposit,
            margin=0.0,
            margin_free=deposit,
            margin_level=np.inf,

            # ---- descriptive ----
            name="John Doe",
            server="MetaTrader5-Simulator",
        )

        self.simulated_mt5.ACCOUNT = sim_ac
        self.DASHBOARD_STATE = DashboardState()
        self.DASHBOARD_STATE.live_data["bot_name"] = self.ea_name
        self.DASHBOARD_STATE.live_data["time"] = start_dt_ts
        self.DASHBOARD_STATE.live_data["balance"] = deposit
        self.DASHBOARD_STATE.live_data["equity"] = deposit
        self.DASHBOARD_STATE.live_data["free_margin"] = deposit

        self.simulated_mt5.DEALS.append(self._make_balance_deal(current_time=self.tester_config["start_date"]))

        return True

    def _initialize_curves(self, tick_size: int):

        self.tester_curves = {
            "time": np.empty(tick_size, dtype=np.int64),
            "balance": np.empty(tick_size, dtype=np.float64),
            "equity": np.empty(tick_size, dtype=np.float64),
            "margin_level": np.empty(tick_size, dtype=np.float64),
        }

    def info_log(self, msg: str):
        if self.IS_OPTIMIZATION_MODE:
            return

        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg, stacklevel=3)

    def debug_log(self, msg: str):
        if self.IS_OPTIMIZATION_MODE:
            return

        if self.logger is None:
            print(msg)
        else:
            self.logger.debug(msg, stacklevel=3)

    def warning_log(self, msg: str):
        if self.IS_OPTIMIZATION_MODE:
            return

        if self.logger is None:
            print(msg)
        else:
            self.logger.warning(msg, stacklevel=3)

    def critical_log(self, msg: str):
        if self.IS_OPTIMIZATION_MODE:
            return

        if self.logger is None:
            print(msg)
        else:
            self.logger.critical(msg, stacklevel=3)

    def error_log(self, msg: str):
        if self.IS_OPTIMIZATION_MODE:
            return

        if self.logger is None:
            print(msg)
        else:
            self.logger.error(msg, stacklevel=3)

    def _get_sim_time(self):
        if self.simulated_mt5 is None:
            return datetime.now(tz=timezone.utc)  # fallback during init

        t = self.simulated_mt5.current_time_msc()
        if t is None:
            return datetime.now(tz=timezone.utc)

        return datetime.fromtimestamp(t / 1000, tz=timezone.utc)

    def _positions_monitoring(self):
        """
        Monitors all open positions and updates the account:
        - updates profit
        - checks SL / TP
        - closes positions when hit
        """

        positions_found = self.simulated_mt5.positions_total()

        self.positions_total_margin = 0
        self.positions_unrealized_pl = 0

        for i in range(positions_found - 1, -1, -1):

            pos = self.simulated_mt5.POSITIONS[i]
            tick = self.simulated_mt5.symbol_info_tick(pos.symbol)

            # --- Determine close price and opposite order type ---
            if pos.type == self.simulated_mt5.POSITION_TYPE_BUY:
                price = tick.bid
                close_type = self.simulated_mt5.ORDER_TYPE_SELL
            elif pos.type == self.simulated_mt5.POSITION_TYPE_SELL:
                price = tick.ask
                close_type = self.simulated_mt5.ORDER_TYPE_BUY
            else:
                self.warning_log("Unknown position type")
                continue

            # --- Update floating profit ---

            profit = self.simulated_mt5.order_calc_profit(
                order_type=pos.type,
                symbol=pos.symbol,
                volume=pos.volume,
                price_open=pos.price_open,
                price_close=price
            )

            self.positions_unrealized_pl += profit
            self.positions_total_margin += pos.margin

            # --- Check SL / TP ---
            hit_tp = False
            hit_sl = False

            if pos.tp > 0:
                hit_tp = (
                    price >= pos.tp if pos.type == self.simulated_mt5.POSITION_TYPE_BUY
                    else price <= pos.tp
                )

            if pos.sl > 0:
                hit_sl = (
                    price <= pos.sl if pos.type == self.simulated_mt5.POSITION_TYPE_BUY
                    else price >= pos.sl
                )

            pos = pos._replace(
                profit=profit,
                price_current=price,
                time_update=tick.time,
                time_update_msc=tick.time_msc
            )

            # MUST write it back
            self.simulated_mt5.POSITIONS[i] = pos

            if not (hit_tp or hit_sl):
                continue

            # --- Close position ---
            request = {
                "action": self.simulated_mt5.TRADE_ACTION_DEAL,
                "type": close_type,
                "symbol": pos.symbol,
                "price": price,
                "volume": pos.volume,
                "position": pos.ticket,
                "comment": "TP hit" if hit_tp else "SL hit",
            }

            self.simulated_mt5.order_send(request)

    def _account_monitoring(self, pos_must_exist: bool = True):

        # ------- monitor the account only if there is at least one position ------

        if (len(self.simulated_mt5.POSITIONS) > 0) if pos_must_exist else True:
            new_equity = self.simulated_mt5.ACCOUNT.balance + self.positions_unrealized_pl
            self.simulated_mt5.ACCOUNT = self.simulated_mt5.ACCOUNT._replace(
                profit=self.positions_unrealized_pl,
                equity=new_equity,
                margin=self.positions_total_margin,
                margin_free=new_equity - self.positions_total_margin,
                margin_level=new_equity / self.positions_total_margin * 100 if self.positions_total_margin > 0 else np.inf
            )

        # ---------- evaluate the margin ---------------------

        margin_evaluation = evaluate_margin_state(self.simulated_mt5.ACCOUNT)
        if margin_evaluation.state == "STOP_OUT":
            self.critical_log(
                f"Account Margin STOPOUT Triggered! Evaluation: {margin_evaluation} balance {self.simulated_mt5.ACCOUNT.balance}, equity: {self.simulated_mt5.ACCOUNT.equity}, margin: {self.simulated_mt5.ACCOUNT.margin}, margin level: {self.simulated_mt5.ACCOUNT.margin_level}")
            self.IS_MARGIN_STOPOUT = True

        if margin_evaluation.state == "MARGIN_CALL":
            self.warning_log(
                f"Account Margin CALL Triggered! Evaluation: {margin_evaluation} balance {self.simulated_mt5.ACCOUNT.balance}, equity: {self.simulated_mt5.ACCOUNT.equity}, margin: {self.simulated_mt5.ACCOUNT.margin}, margin level: {self.simulated_mt5.ACCOUNT.margin_level}")

    def _pending_orders_monitoring(self):

        """
        Monitors pending orders:
        - handles expiration
        - triggers STOP / LIMIT orders correctly
        - converts them into market positions
        """

        for i in reversed(range(len(self.simulated_mt5.ORDERS))):

            order = self.simulated_mt5.ORDERS[i]

            symbol = order.symbol
            tick = self.simulated_mt5.symbol_info_tick(symbol)

            ask = tick.ask
            bid = tick.bid

            # ---------------- UPDATE price_current ----------------

            if order.type in self.simulated_mt5.BUY_ACTIONS:
                new_price_current = ask
                final_pos_type = self.simulated_mt5.POSITION_TYPE_BUY
            else:
                new_price_current = bid
                final_pos_type = self.simulated_mt5.POSITION_TYPE_SELL

            updated_order = order._replace(price_current=new_price_current)  # price mod ASAP
            self.simulated_mt5.ORDERS[i] = updated_order

            order = updated_order

            # --- Expiration handling ---

            expiration_time = order.time_expiration
            if expiration_time and self.simulated_mt5.current_time >= expiration_time:
                request = {
                    "action": self.simulated_mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                    "symbol": symbol,
                    "magic": order.magic
                }

                self.simulated_mt5.order_send(request)
                self.simulated_mt5.ORDERS.pop(i)  # safely remove a pending order that expired
                self.debug_log(f"Pending order #{order.ticket} expired!")
                continue

            triggered = False

            order_type = order.type
            order_price = order.price_open

            # -------- BUY ORDERS --------
            if order_type == self.simulated_mt5.ORDER_TYPE_BUY_LIMIT:
                if new_price_current <= order_price:
                    triggered = True

            elif order_type == self.simulated_mt5.ORDER_TYPE_BUY_STOP:
                if new_price_current >= order_price:
                    triggered = True

            # -------- SELL ORDERS --------
            elif order_type == self.simulated_mt5.ORDER_TYPE_SELL_LIMIT:
                if new_price_current >= order_price:
                    triggered = True

            elif order_type == self.simulated_mt5.ORDER_TYPE_SELL_STOP:
                if new_price_current <= order_price:
                    triggered = True

            if not triggered:
                continue

            sl_diff = abs(order.sl - order.price_open)
            tp_diff = abs(order.tp - order.price_open)

            if order.type in self.simulated_mt5.BUY_ACTIONS:
                pos_price = new_price_current
                new_sl = pos_price - sl_diff
                new_tp = pos_price + tp_diff
            else:
                pos_price = new_price_current
                new_sl = pos_price + sl_diff
                new_tp = pos_price - tp_diff

            # ----- Execute pending order -----
            request = {
                "action": self.simulated_mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "type": final_pos_type,
                "price": pos_price,
                "sl": new_sl,
                "tp": new_tp,
                "volume": order.volume_current,
                "magic": order.magic,
                "comment": order.comment,
                "order": order.ticket,  # an additional field to use for tracking history of this order
            }

            self.simulated_mt5.order_send(request)
            self.debug_log(f"Pending order #{order.ticket} triggered into a position!")
            self.simulated_mt5.ORDERS.pop(i)  # safely remove a pending order that becomes a position | TRIGGERED

    def _monitor_mt5(self,
                     current_time: int,
                     dashboard_fps: int = 60) -> None:

        """
            Monitors the simulated MetaTrader5 instance by updating virtual position, orders, and account credentials

            Args:
                current_time: The current tick time.
                dashboard_fps: The interval to update the chart on a browser alongside the trades table with other active values such as balance, equity, etc.
        """

        # Monitor trading operations

        if self.simulated_mt5.positions_total() > 0:
            self._positions_monitoring()

        if self.simulated_mt5.orders_total() > 0:
            self._pending_orders_monitoring()

        if self.simulated_mt5.orders_total() > 0 or self.simulated_mt5.positions_total() > 0:
            self._account_monitoring()

        # record curves

        if self.simulated_mt5.positions_total() > 0:  # update the curves only if there is atleast one position
            # update curves according to a timeframe

            tf_str = self.tester_config.get("timeframe", "M1")
            tf_int = self.simulated_mt5.STRING2TIMEFRAME_MAP[tf_str]

            if current_time % PeriodSeconds(tf_int) == 0:

                self._record_curve_point()  # record curve values (bal, eq, margin, etc.)

                if not self.IS_OPTIMIZATION_MODE:  # Dashboard updates, only if we are not in optimization mode

                    ac = self.simulated_mt5.ACCOUNT

                    self.DASHBOARD_STATE.live_data["time"] = self.simulated_mt5.current_time()
                    self.DASHBOARD_STATE.live_data["balance"] = ac.balance
                    self.DASHBOARD_STATE.live_data["equity"] = ac.equity
                    self.DASHBOARD_STATE.live_data["free_margin"] = ac.margin_free

                    order_type_map = {
                        self.simulated_mt5.ORDER_TYPE_BUY: "Buy",
                        self.simulated_mt5.ORDER_TYPE_SELL: "Sell",
                        self.simulated_mt5.ORDER_TYPE_BUY_LIMIT: "Buy Limit",
                        self.simulated_mt5.ORDER_TYPE_SELL_LIMIT: "Sell Limit",
                        self.simulated_mt5.ORDER_TYPE_BUY_STOP: "Buy Stop",
                        self.simulated_mt5.ORDER_TYPE_SELL_STOP: "Sell Stop"
                    }

                    positions = [
                        {
                            **pos._asdict(),
                            "type": order_type_map[pos.type]
                        }

                        for pos in self.simulated_mt5.POSITIONS
                    ]

                    orders = [
                        {
                            **order._asdict(),
                            "type": order_type_map[order.type]
                        }

                        for order in self.simulated_mt5.ORDERS
                    ]

                    # combine
                    trades = positions + orders

                    # sort so positions first, pending orders last
                    trades.sort(key=lambda x: x["type"])

                    # if not self.IS_MARGIN_STOPOUT: # if margin stopout is reached, prevent updating the trades dashboard just for reference
                    self.DASHBOARD_STATE.live_data["trades"] = trades

                    # live dashboard update

                    dashboard_interval = 1.0 / dashboard_fps
                    now = time.perf_counter()

                    if now - self._last_dashboard_update >= dashboard_interval:
                        self._last_dashboard_update = now

                        socketio.emit(
                            "dashboard_update",
                            self.DASHBOARD_STATE.live_data
                        )

                        socketio.sleep(0)

    def _run_tick_simulation(
            self,
            df: pl.DataFrame,
            symbols: list[str],
            on_tick_function,
            dashboard_fps: int = 60
    ):
        """
        This function Drives the strategy tester using grouped tick events.

        Args:
            df (pl.DataFrame): A Polars DataFrame containing tick data, with columns for time, symbol_id, bid, ask, etc.
            symbols (list[str]): A list mapping symbol_id to actual symbol names.
            on_tick_function: A callback function that executes the strategy logic on each tick. This function is called after all ticks for a given timestamp are processed and the simulated MetaTrader5 instance is updated with the latest tick information.
            dashboard_fps: The interval to update the chart on a browser alongside the trades table with other active values such as balance, equity, etc.
        """

        total_rows = df.height
        processed = 0

        grouped = df.group_by("time_msc", maintain_order=True)
        with tqdm(total=total_rows, desc="StrategyTester Progress", unit="tick") as pbar:

            for _, rows in grouped:
                n = rows.height

                # process all ticks at this timestamp
                for row in rows.iter_rows(named=True):
                    symbol = symbols[row["symbol_id"]]
                    self.simulated_mt5.tick_update(symbol, row)

                if self.IS_MARGIN_STOPOUT:
                    break

                t = row["time"]
                self._monitor_mt5(current_time=t, dashboard_fps=dashboard_fps)

                # update progress AFTER processing group
                processed += n
                self.TESTER_IDX += 1  # increment tester progress

                pbar.update(n)

                # call strategy AFTER all symbols updated
                on_tick_function()

    def _run_bar_simulation(
            self,
            df: pl.DataFrame,
            symbols: list[str],
            on_tick_function,
            dashboard_fps: int = 60
    ):

        """
        This function Drives the strategy tester using grouped bars.

        Args:
            df (pl.DataFrame): A Polars DataFrame containing bars data, with columns for time, open, high, low, close, etc.
            symbols (list[str]): A list mapping symbol_id to actual symbol names.
            on_tick_function: A callback function that executes the strategy logic on each tick. This function is called after all ticks for a given timestamp are processed and the simulated MetaTrader5 instance is updated with the latest tick information.
            dashboard_fps: The interval to update the chart on a browser alongside the trades table with other active values such as balance, equity, etc.
        """

        total_rows = df.height
        processed = 0

        grouped = df.group_by("time", maintain_order=True)
        with tqdm(total=total_rows, desc="StrategyTester Progress", unit="bar") as pbar:

            current_time = self.tester_config["start_date"]

            for _, rows in grouped:
                n = rows.height

                # process all ticks at this timestamp
                for row in rows.iter_rows(named=True):
                    symbol = symbols[row["symbol_id"]]

                    s_info = self.simulated_mt5.symbol_info(symbol)

                    if s_info is None:
                        self.critical_log("No symbol info found in the simulated (virtual) MetaTrader5 instance")
                        continue

                    current_time = row["time"]

                    tick = Tick(
                        time=current_time,
                        bid=row["close"],
                        ask=row["close"] + row["spread"] * s_info.point,
                        last=0,
                        volume=row["tick_volume"],
                        time_msc=row["time"] * 1000,
                        flags=-1,
                        volume_real=0
                    )

                    self.simulated_mt5.tick_update(symbol, tick)

                if self.IS_MARGIN_STOPOUT:
                    break

                self._monitor_mt5(current_time=current_time, dashboard_fps=dashboard_fps)

                # update progress AFTER processing group
                processed += n
                self.TESTER_IDX += 1  # increment tester progress

                pbar.update(n)

                # call strategy AFTER all symbols updated
                on_tick_function()

    def run(self,
            on_tick_function: Any,
            is_optimization_mode: bool = False,
            dashboard_host: str = "localhost",
            dashboard_port: int = 5000,
            dashboard_fps: int = 60
            ) -> Optional[stats.TesterStats]:

        """Main function to run the strategy tester simulation. It initializes the tester, processes historical data according to the specified modelling mode, and generates a report at the end.

        Args:
            on_tick_function: A callback function that executes the strategy logic on each tick. This function is called after all ticks for a given timestamp are processed and the simulated MetaTrader5 instance is updated with the latest tick information.
            is_optimization_mode: When set to true, it runs the simulator and everything in "light-mode". It prevents logging, real-time dashboard, avoids unnecessary calculations. Just to end up with a smooth backtest.
            dashboard_host : The local server host
            dashboard_port : The local server port
            dashboard_fps: The interval to update the chart on a browser alongside the trades table with other active values such as balance, equity, etc.

        Returns:
            TesterStats (optional): An object containing various statistics computed from the tester results, including trade performance metrics, drawdowns, and more. This is the same stats object that is used to generate the final HTML report.
        """

        self.IS_OPTIMIZATION_MODE = is_optimization_mode

        # ---------------- START SERVER ----------------
        def start_dashboard():

            if not webbrowser.open(f"http://{dashboard_host}:{dashboard_port}"):
                self.warning_log("Failed to open browser for dashboard")

            socketio.run(
                app,
                host=dashboard_host,
                port=dashboard_port,
                debug=False,
                use_reloader=False,
                allow_unsafe_werkzeug=True
            )

        if not is_optimization_mode:
            # run flask in background
            threading.Thread(
                target=start_dashboard,
                daemon=True,
            ).start()

            # time.sleep(1)

        start_date = self.tester_config["start_date"]
        end_date = self.tester_config["end_date"]
        symbols = self.tester_config["symbols"]
        modelling = self.tester_config["modelling"]
        timeframe = self.tester_config["timeframe"]

        # synchronize the data once during optimization

        if self.IS_OPTIMIZATION_FIRST_RUN:
            self.history_manager.synchronize_all_timeframes(symbols, start_date, end_date)

            if not self._tester_init(is_optimization_mode=self.IS_OPTIMIZATION_MODE):  # initialize the tester
                self.error_log("Failed to initialize StrategyTester")
                return None

        if modelling == 4:

            # build tick stream

            if self.IS_OPTIMIZATION_FIRST_RUN:
                self.history_dataframe = self.history_manager.build_tick_stream(
                    symbols,
                    start_date,
                    end_date,
                    True,
                    self.polars_collect_engine
                )

            self._initialize_curves(tick_size=self.history_dataframe.height)

            # run simulation
            self._run_tick_simulation(
                self.history_dataframe,
                symbols,
                on_tick_function,
                dashboard_fps=dashboard_fps
            )

        elif modelling in (2, 1):

            if not self._tester_init(is_optimization_mode=self.IS_OPTIMIZATION_MODE):  # initialize the tester
                self.error_log("Failed to initialize StrategyTester")
                return

            if self.IS_OPTIMIZATION_FIRST_RUN:
                self.history_dataframe = self.history_manager.build_bar_stream(
                    symbols,
                    self.simulated_mt5.STRING2TIMEFRAME_MAP["M1"] if modelling == 1 else
                    self.simulated_mt5.STRING2TIMEFRAME_MAP[timeframe],
                    start_date,
                    end_date,
                    True,
                    self.polars_collect_engine
                )

            self._initialize_curves(tick_size=self.history_dataframe.height)

            # run bar simulation
            self._run_bar_simulation(
                self.history_dataframe,
                symbols,
                on_tick_function,
                dashboard_fps=dashboard_fps
            )

        # ---------------------- END OF THE TEST ---------------------
        # terminate all open positions

        cmt = "Margin stopout" if self.IS_MARGIN_STOPOUT else "End of test"
        self.simulated_mt5._terminate_all_positions(comment=cmt)

        self._record_curve_point()
        tester_stats = self.generate_tester_stats()  # Generate tester stats

        if not self.IS_OPTIMIZATION_MODE:

            # Save trading history (orders & deals) into separate CSV files
            self._save_trading_history(self.trading_history_dir)

            # introduce backtest report to the dashboard
            self.DASHBOARD_STATE.simulation_running = False

            tester_stats_dict = tester_stats.to_dict() if tester_stats else {}
            if tester_stats:
                self.DASHBOARD_STATE.tester_stats = tester_stats_dict

            self.DASHBOARD_STATE.entries_pl_plot = self.generate_entries_pl_plot_json(
                deals_df=pd.DataFrame(self.simulated_mt5.DEALS),
            )

            holding = self.generate_holding_time_json(
                orders_df=pd.DataFrame(self.simulated_mt5.ORDERS_HISTORY),
            )

            self.DASHBOARD_STATE.holding_plot = holding.get("plot", {})
            self.DASHBOARD_STATE.holding_stats = holding.get("summary", {})

            payload = {
                "tester_stats": tester_stats_dict,
                "holding_stats": self.DASHBOARD_STATE.holding_stats,
                "entries_plot": self.DASHBOARD_STATE.entries_pl_plot,
                "holding_plot": self.DASHBOARD_STATE.holding_plot
            }

            socketio.emit(
                "simulation_finished",
                payload
            )

            time.sleep(1)

        else:
            self.IS_OPTIMIZATION_FIRST_RUN = False
            self.simulated_mt5.reset_state()  # reset data in the simulated MetaTrader5 instance

        return tester_stats

    def generate_tester_stats(self):

        curves = self.tester_curves
        n = int(self.CURVES_IDX)

        if n <= 0:
            self.critical_log("It seems no iterations looped through in the method `run()`")
            return None

        # t = curves["time"][:n]
        bal = curves["balance"][:n]
        eq = curves["equity"][:n]
        margin_level = curves["margin_level"][:n]

        return stats.TesterStats(
            deals=self.simulated_mt5.DEALS,
            initial_deposit=self.tester_config.get("deposit"),
            symbols=len(self.tester_config.get("symbols")),
            balance_curve=bal,
            equity_curve=eq,
            margin_level_curve=margin_level,
            ticks=self.TESTER_IDX,
        )

    def _make_balance_deal(self, current_time: datetime) -> TradeDeal:

        time_sec = int(current_time.timestamp())
        time_msc = int(current_time.timestamp() * 1000)

        return TradeDeal(
            ticket=self.simulated_mt5._generate_deal_ticket(),
            order=0,
            time=time_sec,
            time_msc=time_msc,
            type=self.simulated_mt5.DEAL_TYPE_BALANCE,
            entry=self.simulated_mt5.DEAL_ENTRY_IN,
            magic=0,
            position_id=0,
            reason=np.nan,
            volume=np.nan,
            price=np.nan,
            commission=0.0,
            swap=0.0,
            profit=0.0,
            fee=0.0,
            symbol="",
            balance=self.simulated_mt5.ACCOUNT.balance,
            comment="",
            external_id=""
        )

    def _record_curve_point(self):

        idx = self.CURVES_IDX

        if idx >= len(self.tester_curves["time"]):
            return  # safety guard

        acct = self.simulated_mt5.ACCOUNT

        self.tester_curves["time"][idx] = self.simulated_mt5.current_time()
        self.tester_curves["balance"][idx] = acct.balance
        self.tester_curves["equity"][idx] = acct.equity
        self.tester_curves["margin_level"][idx] = acct.margin_level

        self.CURVES_IDX += 1

    def _save_trading_history(self, path: str):

        # save the trading history
        hist_dir = Path(path)
        hist_dir.mkdir(parents=True, exist_ok=True)

        orders_hist = self.simulated_mt5.ORDERS_HISTORY
        deals_hist = self.simulated_mt5.DEALS

        try:
            orders_csv = hist_dir / "orders.csv"
            pd.DataFrame(orders_hist).to_csv(orders_csv, index=False)

            deals_csv = hist_dir / "deals.csv"
            pd.DataFrame(deals_hist).to_csv(deals_csv, index=False)

        except Exception as e:
            self.warning_log(f"Failed to save trading history: {e!r}")

    @staticmethod
    def generate_entries_pl_plot_json(deals_df: pd.DataFrame) -> str:
        """
         Generates plots for entries and profit & Loss by hours, weekdays, etc.

         Returns:
            HTML reports.
        """
        weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        # ------ calculators -------

        entries = stats.EntriesCalculator(deals_df)
        profit_loss = stats.PLCalculator(deals_df)

        entries_hour = entries.by_hour()
        entries_wd = entries.by_weekday()
        entries_mon = entries.by_month().reindex(range(1, 13), fill_value=0)

        entries_wd.index = weekday_order
        entries_mon.index = month_order

        p_hour, l_hour = profit_loss.profit_by_hour(), profit_loss.loss_by_hour()
        p_wd, l_wd = profit_loss.profit_by_weekday(), profit_loss.loss_by_weekday()
        p_mon, l_mon = profit_loss.profit_by_month(), profit_loss.loss_by_month()

        p_wd.index = weekday_order
        l_wd.index = weekday_order
        p_mon.index = month_order
        l_mon.index = month_order

        # ---- plot ----
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                "Entries by hours",
                "Entries by weekdays",
                "Entries by months",
                "Profit & loss by hours",
                "Profit & loss by weekdays",
                "Profit & loss by months"
            )
        )

        # Row 1: Entries

        fig.add_trace(go.Bar(x=list(entries_hour.index), y=entries_hour.values, name="Entries"),
                      row=1, col=1)

        fig.add_trace(go.Bar(x=list(entries_wd.index), y=entries_wd.values, name="Entries", showlegend=False),
                      row=1, col=2)

        fig.add_trace(go.Bar(x=list(entries_mon.index), y=entries_mon.values, name="Entries", showlegend=False),
                      row=1, col=3)

        # Row 2: Profit & Loss (side-by-side like matplotlib version)

        fig.add_trace(go.Bar(x=[str(i) for i in range(24)], y=p_hour.values, name="Profit"),
                      row=2, col=1)
        fig.add_trace(go.Bar(x=[str(i) for i in range(24)], y=l_hour.values, name="Loss"),
                      row=2, col=1)

        fig.add_trace(go.Bar(x=weekday_order, y=p_wd.values, name="Profit", showlegend=False),
                      row=2, col=2)
        fig.add_trace(go.Bar(x=weekday_order, y=l_wd.values, name="Loss", showlegend=False),
                      row=2, col=2)

        fig.add_trace(go.Bar(x=month_order, y=p_mon.values, name="Profit", showlegend=False),
                      row=2, col=3)
        fig.add_trace(go.Bar(x=month_order, y=l_mon.values, name="Loss", showlegend=False),
                      row=2, col=3)

        fig.update_layout(

            barmode="group",

            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",

            font=dict(
                color="white",
                family="Inter"
            ),

            margin=dict(
                l=40,
                r=20,
                t=60,
                b=40
            ),

            showlegend=False,
        )

        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(
                color="#8ec8ff"
            )
        )

        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(
                color="#8ec8ff"
            )
        )

        return fig.to_json()

    @staticmethod
    def generate_holding_time_json(orders_df: pd.DataFrame) -> Dict:

        if orders_df.empty:
            return {}

        try:
            entry = orders_df["time_setup"]
            exit_ = orders_df["time_done"]
        except KeyError as e:
            print(e)
            return {}

        mask = (
                entry.notna()
                & exit_.notna()
                & (entry > 0)
                & (exit_ > 0)
        )

        durations_minutes = (exit_[mask] - entry[mask]).abs() / 60.0

        if durations_minutes.empty:
            return {}

        bins = [
            0,
            5,
            15,
            60,
            240,
            1440,
            10080,
            43200,
            np.inf
        ]

        labels = [
            "0-5m",
            "5-15m",
            "15m-1h",
            "1-4h",
            "4-24h",
            "1-7d",
            "7d-1mon",
            ">1mon"
        ]

        bucket = pd.cut(
            durations_minutes,
            bins=bins,
            labels=labels,
            right=False
        )

        counts = (
            bucket
            .value_counts()
            .reindex(labels, fill_value=0)
        )

        desc = durations_minutes.describe()

        fig = go.Figure()

        fig.add_trace(

            go.Pie(
                labels=labels,
                values=counts.values,
                hole=0.55,
                textinfo="percent+label",

                marker=dict(
                    colors=[
                        "#1e90ff",
                        "#3ea0ff",
                        "#62b4ff",
                        "#7cc4ff",
                        "#4dff88",
                        "#ffc857",
                        "#ff9966",
                        "#ff5e7d"
                    ]
                )
            )
        )

        fig.update_layout(
            # title="Holding Time Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",

            font=dict(
                color="white",
                family="Inter"
            ),

            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            ),

            showlegend=True,

            legend=dict(
                orientation="v",
                y=1.05
            )
        )

        return {
            "plot": fig.to_json(),
            "summary": {
                "mean": str(pd.to_timedelta(desc["mean"], unit="m")),
                "std": str(pd.to_timedelta(desc["std"], unit="m")),
                "min": str(pd.to_timedelta(desc["min"], unit="m")),
                "q25": str(pd.to_timedelta(desc["25%"], unit="m")),
                "median": str(pd.to_timedelta(desc["50%"], unit="m")),
                "q75": str(pd.to_timedelta(desc["75%"], unit="m")),
                "max": str(pd.to_timedelta(desc["max"], unit="m"))
            }
        }


_optimization_cache: dict[int, StrategyTester] = {}


def run_backtesting(main_function: Any,
                    tester_config: dict,
                    virtual_mt5: VirtualMetaTrader5,
                    is_optimization_mode: bool = False,
                    dashboard_host: str = "localhost",
                    dashboard_port: int = 5000,
                    dashboard_fps: int = 30,
                    logging_level: int = logging.WARNING,
                    logs_dir: Optional[str] = "Logs",
                    trading_history_dir: Optional[str] = "TradesHistory",
                    polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto"
                    ) -> TesterStats:
    """
        Runs the main_function (OnTick) through multiple ticks or bars in history, depending on the type of modelling specified in the tester_config dictionary.

        Args:
            tester_config (dict): Dictionary of tester configuration values.
            virtual_mt5 (MetaTrader5): Virtual (simulated) MetaTrader5 instance.
            logging_level: Minimum severity of messages to record. Uses standard `logging` levels (e.g., logging.DEBUG, INFO, WARNING, ERROR, CRITICAL). Messages below this level are ignored.
            logs_dir (str): Directory for log files.
            trading_history_dir (str | optional) A directory to keep trading history.
            main_function: A callback function that executes the strategy logic on each tick. This function is called after all ticks for a given timestamp are processed and the simulated MetaTrader5 instance is updated with the latest tick information.
            is_optimization_mode: When set to true, it runs the simulator and everything in "light-mode". It prevents logging, real-time dashboard, avoids unnecessary calculations. Just to end up with a quick and smooth backtest which suits multiple backtesting iterations (optimization).
            dashboard_host : The local server host
            dashboard_port : The local server port
            dashboard_fps: The interval to update the chart on a browser alongside the trades table with other active values such as balance, equity, etc.

            polars_collect_engine (str): Engine used by Polars when collecting historical data in functions for obtaining ticks — copy_ticks*, and bars information/rates (copy_rates*). Supported values are:
                - ``"auto"`` (default): Use Polars’ standard in-memory engine and
                    respect the ``POLARS_ENGINE_AFFINITY`` environment variable if set.
                - ``"in-memory"``: Explicitly use the default in-memory engine,
                    optimized with multi-threading and SIMD over Arrow data.
                - ``"streaming"``: Process queries in batches, enabling
                    larger-than-RAM datasets.
                - ``"gpu"``: Use NVIDIA GPUs via RAPIDS cuDF for accelerated execution.
                    Requires installing Polars with GPU support, e.g.:
                    ``pip install polars[gpu] --extra-index-url=https://pypi.nvidia.com``.
        Raises:
            RuntimeError: If required MT5 account info cannot be obtained.

        Returns:
            TesterStats (optional): An object containing various statistics computed from the tester results, including trade performance metrics, drawdowns, and more. This is the same stats object that is used to generate the final HTML report.
    """

    if not isinstance(virtual_mt5, VirtualMetaTrader5):
        raise RuntimeError("virtual_mt5 argument should have the virtualMetaTrader5 instance object.")

    cache_key = id(virtual_mt5)  # one tester per VirtualMT5 instance

    if is_optimization_mode and cache_key in _optimization_cache:
        tester = _optimization_cache[cache_key]
    else:

        tester = StrategyTester(
            tester_config=tester_config,
            virtual_mt5=virtual_mt5,
            logging_level=logging_level,
            logs_dir=logs_dir,
            trading_history_dir=trading_history_dir,
            polars_collect_engine=polars_collect_engine
        )

        if is_optimization_mode:
            _optimization_cache[cache_key] = tester

    return tester.run(
        on_tick_function=main_function,
        is_optimization_mode=is_optimization_mode,
        dashboard_host=dashboard_host,
        dashboard_port=dashboard_port,
        dashboard_fps=dashboard_fps,
    )


def clear_optimization_cache(virtual_mt5: Optional[VirtualMetaTrader5] = None):
    """Call this when the optimization loop is done to free memory."""
    if virtual_mt5 is not None:
        _optimization_cache.pop(id(virtual_mt5), None)
    else:
        _optimization_cache.clear()