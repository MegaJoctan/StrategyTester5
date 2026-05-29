from __future__ import annotations
from typing import Any, Literal, Optional

import pandas as pd
import numpy as np
import polars as pl

import threading
import time

from flask import Flask, jsonify, render_template
import webbrowser

from strategytester5.MetaTrader5 import evaluate_margin_state, TradeDeal, Tick
from strategytester5.MetaTrader5.api import OverLoadedMetaTrader5API
from strategytester5.config_validators import TesterConfigValidators
from strategytester5.MetaTrader5.data import HistoryManager
from strategytester5.MQL5.functions import PeriodSeconds
from . import *
from datetime import datetime
import os
import numpy as np
from . import _html_templates as templates
from . import stats
import logging
from tqdm import tqdm
# import time

import matplotlib as mpl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pathlib import Path

mpl.rcParams["agg.path.chunksize"] = 10000
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 0.9

dashboard_data = {
    "time": np.nan,
    "balance": np.nan,
    "equity": np.nan,
    "free_margin": np.nan,
    "trades": []
}

app = Flask(
    __name__,
    template_folder="../strategytester5/templates"
)

TESTER_STATS = None
SIMULATION_RUNNING = True

# ------------- FLASK ROUTES ---------------

@app.route("/")
def index():
    return render_template("dashboard.html",
                           simulation_running=SIMULATION_RUNNING,
                           live_data=dashboard_data,
                           tester_stats=TESTER_STATS,
                           )

@app.route("/dashboard")
def dashboard():
    return jsonify(dashboard_data)

class StrategyTester:
    """
        The "engine" that drives the entire backtesting process, simulating the MetaTrader5 environment and allowing you to test your trading strategies against historical data. The main method is `run()`, which takes a callback function that contains your strategy logic and executes it on each tick or bar, depending on the modelling mode you choose.

        > Similar to the MetaTrader5 strategy tester
    """

    def __init__(self,
                 tester_config: dict,
                 mt5_instance: Any,
                 logging_level: int = logging.WARNING,
                 logs_dir: Optional[str] = "Logs",
                 reports_dir: Optional[str] = "Reports",
                 history_dir: Optional[str] = "History",
                 trading_history_dir: Optional[str] = "TradingHistory",
                 polars_collect_engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto"):

        """Instantiates the StrategyTester with the given configuration, sets up the simulated MetaTrader5 environment, and prepares for running the backtest.

        Args:
            tester_config (dict): Dictionary of tester configuration values.
            mt5_instance (MetaTrader5): MetaTrader5 API/client instance used for obtaining crucial information from the broker as an attempt to mimic the terminal.
            logging_level: Minimum severity of messages to record. Uses standard `logging` levels (e.g., logging.DEBUG, INFO, WARNING, ERROR, CRITICAL). Messages below this level are ignored.
            logs_dir (str): Directory for log files.
            reports_dir (str): Directory for HTML reports and assets.
            history_dir (str): Directory for historical data storage.
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

        self.reports_dir = reports_dir
        self.history_dir = history_dir
        self.polars_collect_engine = polars_collect_engine
        self.trading_history_dir = trading_history_dir

        # ---------------- validate all configs from a dictionary -----------------

        self.tester_config = TesterConfigValidators.parse_tester_configs(tester_config)

        # -------------------- initialize the Loggers ----------------------------

        self.ea_name = self.tester_config["bot_name"]
        os.makedirs(logs_dir, exist_ok=True)

        self.logger = get_logger(task_name=self.ea_name, logfile=os.path.join(logs_dir, f"{LOG_DATE}.log"),
                                 level=logging_level, time_provider=self._get_sim_time)

        self.live_mt5_instance = mt5_instance
        if self.live_mt5_instance is None:
            raise RuntimeError(
                "Fatal, A live MetaTrader5 Instance isn't given. If you haven't installed it (WINDOWS-ONLY) run `pip install metatrader5`")

        self.broker_data_dir = self.live_mt5_instance.account_info().server

        start_dt = self.tester_config.get("start_date", 0)
        start_dt_ts = start_dt.timestamp() if isinstance(start_dt, datetime) else start_dt

        self.simulated_mt5 = OverLoadedMetaTrader5API(logger=self.logger,
                                                      broker_data_path=self.broker_data_dir,
                                                      polars_collect_engine=polars_collect_engine,
                                                      history_start_date=start_dt,
                                                      live_mt5=self.live_mt5_instance)

        self.simulated_mt5._current_time = start_dt_ts
        self.simulated_mt5._current_time_msc = start_dt_ts * 1000
        self.logger.info("Initialized")
        deposit = self.tester_config["deposit"]

        self.simulated_mt5.ACCOUNT = self.simulated_mt5.ACCOUNT._replace(
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

        dashboard_data["time"] = start_dt_ts
        dashboard_data["balance"] = deposit
        dashboard_data["equity"] = deposit
        dashboard_data["free_margin"] = deposit

        self.logger.debug(f"Simulated account info: {self.simulated_mt5.ACCOUNT}")

        self.positions_unrealized_pl = 0
        self.positions_total_margin = 0

        # -------------------- tester reports ----------------------------

        self.tester_curves = {
            "time": np.array([]),
            "balance": np.array([]),
            "equity": np.array([]),
            "margin_level": np.array([])
        }

        self.TESTER_IDX = 0
        self.CURVES_IDX = 0
        self.IS_STOPPED = False
        # self._engine_lock = threading.RLock()   # re-entrant lock (safe if functions call other locked functions)

        # ---------------------- others ------------------------------

        self.last_tick_time: int = 0

    @staticmethod
    def _find_mt5_executable(installation_path: str) -> tuple[str, str]:
        """
        Scan a folder and return the MT5 terminal executable (first one containing 'terminal').
        """
        for entry in os.scandir(installation_path):
            if entry.is_file() and entry.name.lower().endswith(".exe"):
                if "terminal" in entry.name.lower():
                    return entry.name, entry.path

        raise FileNotFoundError(f"No MT5 terminal executable found in {installation_path}")

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
                self.logger.warning("Unknown position type")
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
            self.logger.critical("Account Margin STOPOUT Triggered!")
            self.logger.debug(margin_evaluation)
            # self.logger.debug(f"balance {self.simulated_mt5.ACCOUNT.balance}, equity: {self.simulated_mt5.ACCOUNT.equity}, margin: {self.simulated_mt5.ACCOUNT.margin}, margin level: {self.simulated_mt5.ACCOUNT.margin_level}")
            self.IS_STOPPED = True

        """
        if margin_evaluation.state == "MARGIN_CALL":
            self.logger.critical("Account Margin CALL Triggered!")
            self.logger.debug(margin_evaluation)
            # self.logger.debug(f"balance {self.simulated_mt5.ACCOUNT.balance}, equity: {self.simulated_mt5.ACCOUNT.equity}, margin: {self.simulated_mt5.ACCOUNT.margin}, margin level: {self.simulated_mt5.ACCOUNT.margin_level}")
            self.IS_STOPPED = True
        """

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
                self.logger.debug(f"Pending order #{order.ticket} expired!")
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
            self.logger.debug(f"Pending order #{order.ticket} triggered into a position!")
            self.simulated_mt5.ORDERS.pop(i)  # safely remove a pending order that becomes a position | TRIGGERED

    def _monitor_mt5(self, time: int):

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

            if time % PeriodSeconds(tf_int) == 0:
                self._record_curve_point()

    def run_tick_simulation(
            self,
            df: pl.DataFrame,
            symbols: list[str],
            on_tick_function,
    ):
        """
        This function Drives the strategy tester using grouped tick events.

        Args:
            df (pl.DataFrame): A Polars DataFrame containing tick data, with columns for time, symbol_id, bid, ask, etc.
            symbols (list[str]): A list mapping symbol_id to actual symbol names.
            on_tick_function: A callback function that executes the strategy logic on each tick. This function is called after all ticks for a given timestamp are processed and the simulated MetaTrader5 instance is updated with the latest tick information.
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

                t = row["time"]
                self._monitor_mt5(time=t)

                # update dashboard's data

                ac = self.simulated_mt5.ACCOUNT

                dashboard_data["balance"] = ac.balance
                dashboard_data["equity"] = ac.equity
                dashboard_data["free_margin"] = ac.margin_free

                dashboard_data["positions"] = [
                    pos._asdict()
                    for pos in self.simulated_mt5.POSITIONS
                ]

                dashboard_data["orders"] = [
                    order._asdict()
                    for order in self.simulated_mt5.ORDERS
                ]

                # update progress AFTER processing group
                processed += n
                self.TESTER_IDX += 1  # increment tester progress

                pbar.update(n)

                # call strategy AFTER all symbols updated
                on_tick_function()

    def run_bar_simulation(
            self,
            df: pl.DataFrame,
            symbols: list[str],
            on_tick_function,
    ):

        """
        This function Drives the strategy tester using grouped bars.

        Args:
            df (pl.DataFrame): A Polars DataFrame containing bars data, with columns for time, open, high, low, close, etc.
            symbols (list[str]): A list mapping symbol_id to actual symbol names.
            on_tick_function: A callback function that executes the strategy logic on each tick. This function is called after all ticks for a given timestamp are processed and the simulated MetaTrader5 instance is updated with the latest tick information.
        """

        total_rows = df.height
        processed = 0

        grouped = df.group_by("time", maintain_order=True)
        with tqdm(total=total_rows, desc="StrategyTester Progress", unit="bar") as pbar:

            for _, rows in grouped:
                n = rows.height

                # process all ticks at this timestamp
                for row in rows.iter_rows(named=True):
                    symbol = symbols[row["symbol_id"]]
                    point = self.simulated_mt5.symbol_info(symbol).point
                    t = row["time"]

                    tick = Tick(
                        time=t,
                        bid=row["close"],
                        ask=row["close"] + row["spread"] * point,
                        last=0,
                        volume=row["tick_volume"],
                        time_msc=row["time"] * 1000,
                        flags=-1,
                        volume_real=0
                    )

                    self.simulated_mt5.tick_update(symbol, tick)

                self._monitor_mt5(time=t)

                # update dashboard's data

                ac = self.simulated_mt5.ACCOUNT

                dashboard_data["time"] = t
                dashboard_data["balance"] = ac.balance
                dashboard_data["equity"] = ac.equity
                dashboard_data["free_margin"] = ac.margin_free

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

                        "time": datetime
                        .fromtimestamp(pos.time)
                        .strftime("%Y-%m-%d %H:%M:%S"),
                        "type": order_type_map[pos.type]
                    }

                    for pos in self.simulated_mt5.POSITIONS
                ]

                orders = [
                    {
                        **order._asdict(),

                        "time_setup": datetime
                        .fromtimestamp(order.time_setup)
                        .strftime("%Y-%m-%d %H:%M:%S"),
                        "type": order_type_map[order.type]
                    }

                    for order in self.simulated_mt5.ORDERS
                ]

                # combine
                trades = positions + orders

                # sort so positions first, pending orders last
                trades.sort(key=lambda x: x["type"])

                dashboard_data["trades"] = trades

                # print(dashboard_data)

                # update progress AFTER processing group
                processed += n
                self.TESTER_IDX += 1  # increment tester progress

                pbar.update(n)

                # call strategy AFTER all symbols updated
                on_tick_function()

    def run(self,
            on_tick_function: Any,
            dashboard_host: str="localhost",
            dashboard_port: int=5000) -> stats.TesterStats:

        """Main function to run the strategy tester simulation. It initializes the tester, processes historical data according to the specified modelling mode, and generates a report at the end.

        Args:
            on_tick_function: A callback function that executes the strategy logic on each tick. This function is called after all ticks for a given timestamp are processed and the simulated MetaTrader5 instance is updated with the latest tick information.
            dashboard_host : The local server host
            dashboard_port : The local server port

        Returns:
            TesterStats: An object containing various statistics computed from the tester results, including trade performance metrics, drawdowns, and more. This is the same stats object that is used to generate the final HTML report.
        """

        # ---------------- START SERVER ----------------

        def start_dashboard():

            # open browser automatically
            webbrowser.open(f"http://{dashboard_host}:{dashboard_port}")

            # use_reloader=False avoids double execution
            app.run(
                host=dashboard_host,
                port=dashboard_port,
                debug=False,
                use_reloader=False
            )

        # ---------- Run a live dashboard ---------------

        # run flask in background
        threading.Thread(
            target=start_dashboard,
            daemon=True
        ).start()

        # small delay so server starts first
        time.sleep(2)

        start_date = self.tester_config["start_date"]
        end_date = self.tester_config["end_date"]
        symbols = self.tester_config["symbols"]
        modelling = self.tester_config["modelling"]
        timeframe = self.tester_config["timeframe"]

        sync = (self.live_mt5_instance is not None)

        history_manager = HistoryManager(
            mt5_instance=self.live_mt5_instance,
            logger=self.logger,
            broker_data_path=self.broker_data_dir
        )

        # optional: keep this if you want auto-sync
        history_manager.synchronize_all_timeframes(
            symbols, start_date, end_date
        )

        if modelling == 4:
            # build tick stream
            df = history_manager.build_tick_stream(
                symbols,
                start_date,
                end_date,
                sync,
                self.polars_collect_engine
            )

            self._tester_init(size=df.height)  # initialize the tester

            # run simulation
            self.run_tick_simulation(
                df,
                symbols,
                on_tick_function,
            )

        elif modelling in (2, 1):

            df = history_manager.build_bar_stream(
                symbols,
                self.simulated_mt5.STRING2TIMEFRAME_MAP["M1"] if modelling == 1 else
                self.simulated_mt5.STRING2TIMEFRAME_MAP[timeframe],
                start_date,
                end_date,
                sync,
                self.polars_collect_engine
            )

            self._tester_init(size=df.height)  # initialize the tester

            # run bar simulation
            self.run_bar_simulation(
                df,
                symbols,
                on_tick_function,
            )

        self._tester_deinit()

        # update the dashboard

        global TESTER_STATS, SIMULATION_RUNNING

        SIMULATION_RUNNING = False
        TESTER_STATS = self.generate_tester_stats()

        return TESTER_STATS

    def generate_tester_stats(self):

        curves = self.tester_curves
        n = int(self.CURVES_IDX)

        if n <= 0:
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

    def _make_balance_deal(self, time: datetime) -> TradeDeal:

        time_sec = int(time.timestamp())
        time_msc = int(time.timestamp() * 1000)

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

    def _tester_init(self, size: int):

        self.TESTER_IDX = 0

        self.tester_curves = {
            "time": np.empty(size, dtype=np.int64),
            "balance": np.empty(size, dtype=np.float64),
            "equity": np.empty(size, dtype=np.float64),
            "margin_level": np.empty(size, dtype=np.float64),
        }

        self.simulated_mt5.DEALS.append(
            self._make_balance_deal(time=self.tester_config["start_date"])
        )

    def _tester_deinit(self):

        # terminate all open positions
        self.simulated_mt5._terminate_all_positions(comment="End of test")

        # Build final curves (base pre-allocated slice + 1 extra point)

        n = int(self.CURVES_IDX)

        if n > 0:
            self.tester_curves["balance"][n - 1] = self.simulated_mt5.ACCOUNT.balance
            self.tester_curves["equity"][n - 1] = self.simulated_mt5.ACCOUNT.balance
            self.tester_curves["margin_level"][n - 1] = self.simulated_mt5.ACCOUNT.margin_level

        # generate a report at the end

        os.makedirs(self.reports_dir, exist_ok=True)

        self._save_trading_history(self.trading_history_dir)

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
            self.logger.warning(f"Failed to save trading history: {e!r}")

    @staticmethod
    def _entries_and_pl_plotly(deals_df: pd.DataFrame):

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
            barmode="group",  # side-by-side exactly like your matplotlib version
            margin=dict(l=80, r=20, t=30, b=40),
            showlegend=False,
        )

        return fig.to_html(
            full_html=False,
            # include_plotlyjs="cdn",
            config={"responsive": True}
        )

    @staticmethod
    def _holding_time_dashboard_figure(orders_df: pd.DataFrame) -> str:
        # --- build durations in minutes (numeric) ---

        if orders_df.empty:
            return ""

        entry = orders_df["time_setup"]
        exit_ = orders_df["time_done"]

        # keep only closed rows (avoid time_done == 0)
        m = entry.notna() & exit_.notna() & (entry > 0) & (exit_ > 0)
        durations_minutes = (exit_[m] - entry[m]).abs() / 60.0

        if durations_minutes.empty:
            fig = go.Figure()
            fig.update_layout(title="No valid closed positions to compute holding time.")
            return ""

        # --- pie buckets ---
        bins = [0, 5, 15, 60, 240, 1440, 10080, 43200, np.inf]  # minutes
        labels = ["0-5m", "5-15m", "15m-1h", "1-4h", "4-24h", "1-7d", "7d-1mon", ">1mon"]

        bucket = pd.cut(durations_minutes, bins=bins, labels=labels, right=False)
        counts = bucket.value_counts().reindex(labels, fill_value=0)

        # --- describe stats for the table ---
        desc = durations_minutes.describe()  # count, mean, std, min, 25%, 50%, 75%, max

        # format values (you can adjust)
        table_header = ["mean", "std", "min", "25%", "50%", "75%", "max"]
        table_values = [
            f"{pd.to_timedelta(desc['mean'], unit='m')}",
            f"{pd.to_timedelta(desc['std'], unit='m')}",
            f"{pd.to_timedelta(desc['min'], unit='m')}",
            f"{pd.to_timedelta(desc['25%'], unit='m')}",
            f"{pd.to_timedelta(desc['50%'], unit='m')}",
            f"{pd.to_timedelta(desc['75%'], unit='m')}",
            f"{pd.to_timedelta(desc['max'], unit='m')}",
        ]

        # --- figure layout: pie (top) + table (bottom) ---
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "table"}]],
            column_widths=[0.6, 0.4],
            vertical_spacing=0.10,
            subplot_titles=("Positions by holding-time bucket", "Holding time summary"),
            horizontal_spacing=0.15  # increase this (default ~0.05)
        )

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=counts.values,  # ensure it's an array
                hole=0.35,
                textinfo="percent+label"
            ),
            row=1, col=1
        )

        # column x rows format: header across top, ONE row of values

        fig.add_trace(go.Table(
            header=dict(values=['Parameter', 'Time']),
            cells=dict(values=[table_header, table_values])
        ),
            row=1, col=2
        )

        fig.update_layout(
            margin=dict(l=40, r=100, t=30, b=30),
            showlegend=False
        )

        return fig.to_html(
            full_html=False,
            # include_plotlyjs="cdn",
            config={"responsive": True}
        )
