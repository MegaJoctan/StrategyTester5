import numpy as np
from numba import njit
from .MetaTrader5.api import MetaTrader5Constants
from scipy.stats import linregress
import pandas as pd
import warnings
import inspect

# np.seterr(all="raise")  # turn numpy warnings into exceptions
# warnings.filterwarnings("error")

@njit(cache=True)
def _max_dd_money_and_pct_nb(x: np.ndarray, eps: float = 1e-12):
    """
    Returns (max_dd_money, max_dd_percent) using MT5 local-high -> next-local-low definition.
    """
    n = x.size
    if n < 3:
        return 0.0, 0.0

    peaks = np.empty(n, dtype=np.int64)
    troughs = np.empty(n, dtype=np.int64)
    pcount = 0
    tcount = 0

    i = 1
    while i < n - 1:
        left = x[i - 1]
        mid = x[i]

        j = i
        while j < n - 1 and abs(x[j] - x[j + 1]) <= eps:
            j += 1
        right = x[j + 1] if j < n - 1 else x[j]

        if mid > left + eps and mid > right + eps:
            peaks[pcount] = i
            pcount += 1
        elif mid < left - eps and mid < right - eps:
            troughs[tcount] = i
            tcount += 1

        i = j + 1

    if pcount == 0 or tcount == 0:
        return 0.0, 0.0

    max_dd = 0.0
    max_pct = 0.0
    t_idx = 0

    for pi in range(pcount):
        p = peaks[pi]
        while t_idx < tcount and troughs[t_idx] <= p:
            t_idx += 1
        if t_idx >= tcount:
            break

        t = troughs[t_idx]
        peak_val = x[p]
        dd = peak_val - x[t]
        if dd > max_dd:
            max_dd = dd

        if peak_val > eps:
            pct = dd / peak_val * 100.0
            if pct > max_pct:
                max_pct = pct

    return max_dd if max_dd > 0.0 else 0.0, max_pct if max_pct > 0.0 else 0.0


class TesterStats:
    """ Computes various statistics from the tester results, including trade performance metrics, drawdowns, and more.

    This class is responsible fo calculating all the stats you see in the HTML report"""

    def __init__(self,
                 deals: list,
                 initial_deposit: float,
                 balance_curve: np.ndarray,
                 equity_curve: np.ndarray,
                 margin_level_curve: np.ndarray,
                 ticks: int,
                 symbols: int
                 ):

        """ Initializes the TesterStats object with the provided data and computes all statistics.

        Args:
            deals (list): List of deal records from the tester.
            initial_deposit (float): The initial deposit amount used in the test.
            balance_curve (np.ndarray): Array representing the balance curve over time.
            equity_curve (np.ndarray): Array representing the equity curve over time.
            margin_level_curve (np.ndarray): Array representing the margin level curve over time.
            ticks (int): Total number of ticks processed during the test.
            symbols (int): Total number of unique symbols traded during the test.
        """

        self.deals = deals
        self.initial_deposit_ = float(initial_deposit)
        self.balance_curve = np.ascontiguousarray(np.asarray(balance_curve, dtype=np.float64)).reshape(-1)
        self.equity_curve = np.ascontiguousarray(np.asarray(equity_curve, dtype=np.float64)).reshape(-1)
        self.margin_level_curve = np.ascontiguousarray(np.asarray(margin_level_curve, dtype=np.float64)).reshape(-1)
        self.ticks_ = ticks
        self.symbols_ = symbols

        self._profits: list[float] = []
        self._losses: list[float] = []  # negative profits (losses)
        self._returns = np.diff(self.equity_curve)

        self._total_trades = 0
        self._total_long_trades = 0
        self._total_short_trades = 0
        self._long_trades_won = 0
        self._short_trades_won = 0

        self._max_consec_win_count = 0
        self._max_consec_win_money = 0.0
        self._max_consec_loss_count = 0
        self._max_consec_loss_money = 0.0

        self._max_profit_streak_money = 0.0
        self._max_profit_streak_count = 0
        self._max_loss_streak_money = 0.0
        self._max_loss_streak_count = 0

        self._win_streaks: list[int] = []
        self._loss_streaks: list[int] = []
        self._trade_returns = []  # per-trade returns in *fraction* (e.g., 0.01 = +1%)

        self.eps = 1e-10

        self._compute()

        y = self.balance_curve.astype(float)
        x = np.arange(len(y), dtype=float)

        if len(x) == 0 or len(y) == 0:
            self.lr_res = None
        else:
            self.lr_res = linregress(x, y)

    def _compute(self):
        cur_win_count = 0
        cur_win_money = 0.0
        cur_loss_count = 0
        cur_loss_money = 0.0

        for d in self.deals:
            if getattr(d, "entry", None) != MetaTrader5Constants.DEAL_ENTRY_OUT:
                continue

            self._total_trades += 1

            d_type = d.type
            if d_type == MetaTrader5Constants.DEAL_TYPE_BUY:
                self._total_long_trades += 1
            elif d_type == MetaTrader5Constants.DEAL_TYPE_SELL:
                self._total_short_trades += 1

            profit = d.profit + d.commission

            # ---- per-trade return (percent change per trade) for AHPR/GHPR :contentReference[oaicite:10]{index=10}
            bal_after = getattr(d, "balance", None)
            if bal_after is not None:
                bal_after = float(bal_after)
                bal_before = bal_after - profit
                if bal_before > self.eps:
                    self._trade_returns.append(profit / bal_before)

            if profit > 0.0:

                self._profits.append(profit)

                if cur_loss_count > 0:
                    self._loss_streaks.append(cur_loss_count)
                    cur_loss_count = 0
                    cur_loss_money = 0.0

                cur_win_count += 1
                cur_win_money += profit

                if cur_win_count > self._max_consec_win_count:
                    self._max_consec_win_count = cur_win_count
                    self._max_consec_win_money = cur_win_money

                if cur_win_money > self._max_profit_streak_money:
                    self._max_profit_streak_money = cur_win_money
                    self._max_profit_streak_count = cur_win_count

                if d_type == MetaTrader5Constants.DEAL_TYPE_BUY:
                    self._long_trades_won += 1
                elif d_type == MetaTrader5Constants.DEAL_TYPE_SELL:
                    self._short_trades_won += 1

            else:
                self._losses.append(profit)  # negative or zero

                if cur_win_count > 0:
                    self._win_streaks.append(cur_win_count)
                    cur_win_count = 0
                    cur_win_money = 0.0

                cur_loss_count += 1
                cur_loss_money += profit  # negative accumulation

                if cur_loss_count > self._max_consec_loss_count:
                    self._max_consec_loss_count = cur_loss_count
                    self._max_consec_loss_money = cur_loss_money

                if cur_loss_money < self._max_loss_streak_money:
                    self._max_loss_streak_money = cur_loss_money
                    self._max_loss_streak_count = cur_loss_count

        # flush last streaks (important!)
        if cur_win_count > 0:
            self._win_streaks.append(cur_win_count)
        if cur_loss_count > 0:
            self._loss_streaks.append(cur_loss_count)

    @property
    def initial_deposit(self):
        """The initial trading capital for the backtest"""
        return self.initial_deposit_

    @property
    def ticks(self):
        """The number of ticks seen during the backtest"""
        return self.ticks_

    @property
    def symbols(self):
        """The number of symbols seen during the backtest"""
        return self.symbols_

    @property
    def total_trades(self) -> int:
        """ Total number of trades opened and closed during the test. """
        return self._total_trades

    @property
    def total_deals(self) -> int:
        """The total number of deal records, including both entries and exits. Note that the first deal is usually the initial deposit and is not a real trade."""
        return len(self.deals) - 1

    @property
    def total_short_trades(self) -> int:
        """ Total number of short (SELL) trades closed during the test. """
        return self._total_short_trades

    @property
    def total_long_trades(self) -> int:
        """ Total number of long (BUY) trades closed during the test. """
        return self._total_long_trades

    @property
    def short_trades_won(self) -> int:
        """ Number of short (SELL) trades that were profitable (profit > 0) at closing. """
        return self._short_trades_won

    @property
    def long_trades_won(self) -> int:
        """ Number of long (BUY) trades that were profitable (profit > 0) at closing. """
        return self._long_trades_won

    @property
    def profit_trades(self) -> int:
        """ Number of trades that were profitable (profit > 0) at closing. """
        return len(self._profits) if self._profits else 0

    @property
    def loss_trades(self) -> int:
        """ Number of trades that were not profitable (profit <= 0) at closing. """
        return len(self._losses) if self._losses else 0

    @property
    def largest_profit_trade(self) -> float:
        """ Largest profit from a single trade. """
        return np.max(self._profits) if self._profits else 0

    @property
    def largest_loss_trade(self) -> float:
        """ Largest loss from a single trade. """
        return np.min(self._losses) if self._losses else 0

    @property
    def average_profit_trade(self) -> float:
        """ Average profit from profitable trades. """
        return np.mean(self._profits) if self._profits else 0

    @property
    def average_loss_trade(self) -> float:
        """ Average loss from unprofitable trades. """
        return np.mean(self._losses) if self._losses else 0

    # ---------- streak metrics ----------

    @property
    def maximum_consecutive_wins_count(self) -> int:
        """ Maximum number of consecutive winning trades. """
        return self._max_consec_win_count

    @property
    def maximum_consecutive_wins_money(self) -> float:
        """ Maximum money won from consecutive winning trades. """
        return self._max_consec_win_money

    @property
    def maximum_consecutive_losses_count(self) -> int:
        """ Maximum number of consecutive losing trades. """
        return self._max_consec_loss_count

    @property
    def maximum_consecutive_losses_money(self) -> float:
        """ Maximum money lost from consecutive losing trades. """
        # show as absolute money if you prefer; MT5 shows total loss (negative) in brackets
        return self._max_consec_loss_money

    @property
    def maximal_consecutive_profit_count(self) -> int:
        """ Maximum number of consecutive profitable trades. """
        return self._max_profit_streak_count

    @property
    def maximal_consecutive_profit_money(self) -> float:
        """ Maximum money won from consecutive profitable trades. """
        return self._max_profit_streak_money

    @property
    def maximal_consecutive_loss_count(self) -> int:
        """ Maximum number of consecutive losing trades. """
        return self._max_loss_streak_count

    @property
    def maximal_consecutive_loss_money(self) -> float:
        """ Maximum money lost from consecutive losing trades. """
        return self._max_loss_streak_money

    @property
    def average_consecutive_wins(self) -> float:
        """ Average profit from consecutive winning trades. """
        return np.mean(self._win_streaks) if self._win_streaks else 0

    @property
    def average_consecutive_losses(self) -> float:
        """ Average loss from consecutive losing trades. """
        return np.mean(self._loss_streaks) if self._loss_streaks else 0

    @property
    def gross_profit(self) -> float:
        """ Total profit from all profitable trades. """
        return np.sum(self._profits) if self._profits else 0.0

    @property
    def gross_loss(self) -> float:
        """ Total loss from all unprofitable trades. """
        return np.sum(self._losses) if self._losses else 0.0

    @property
    def net_profit(self) -> float:
        """ Net profit (gross profit - gross loss). """
        return self.gross_profit - np.abs(self.gross_loss)

    @property
    def profit_factor(self) -> float:
        """ Profit factor (gross profit / gross loss). """
        return self.gross_profit / (self.gross_loss + self.eps)

    @property
    def recovery_factor(self) -> float:
        """ Recovery factor (net profit / maximal equity drawdown). """
        return self.net_profit / (self.equity_drawdown_maximal + self.eps)

    @property
    def expected_payoff(self) -> int:
        """ Expected payoff (net profit / total trades). """
        return (self.net_profit / self.total_trades) if self.total_trades > 0 else 0

    # ---------- drawdowns ----------
    @staticmethod
    def _abs_drawdown(initial: float, curve: np.ndarray) -> float:
        if curve.size == 0:
            return 0.0
        min_val = float(np.min(curve))
        dd = initial - min_val
        return dd if dd > 0.0 else 0.0

    @property
    def balance_drawdown_absolute(self) -> float:
        """ Absolute drawdown for balance curve. """
        # AbsoluteDrawDown = InitialDeposit - MinimalBalance (below initial) :contentReference[oaicite:12]{index=12}
        return self._abs_drawdown(self.initial_deposit_, self.balance_curve)

    @property
    def equity_drawdown_absolute(self) -> float:
        """ Absolute drawdown for equity curve. """
        return self._abs_drawdown(self.initial_deposit_, self.equity_curve)

    @property
    def balance_drawdown_maximal(self) -> float:
        """ MT5 maximal drawdown uses local-high, next local low definition, which can be different from absolute drawdown. """
        return float(_max_dd_money_and_pct_nb(self.balance_curve)[0])

    def _validate_baleq_values(self, value: float) -> float:
        if abs(value) > self.initial_deposit_:
            return np.nan

        return value

    @property
    def balance_drawdown_relative(self) -> float:
        """ MT5 relative drawdown uses local high, next local low (max %)"""
        return self._validate_baleq_values(float(_max_dd_money_and_pct_nb(self.balance_curve)[1]))

    @property
    def equity_drawdown_maximal(self) -> float:
        """ MT5 maximal drawdown uses local-high, next local low definition, which can be different from absolute drawdown. """
        return self._validate_baleq_values(float(_max_dd_money_and_pct_nb(self.equity_curve)[0]))

    @property
    def equity_drawdown_relative(self) -> float:
        """ MT5 relative drawdown uses local high, next local low (max %)"""
        return self._validate_baleq_values(float(_max_dd_money_and_pct_nb(self.equity_curve)[1]))

    @property
    def sharpe_ratio(self) -> float:
        """(Return - 0) / std(Return)"""
        r = np.asarray(self._trade_returns, dtype=np.float64)

        if r.size < 2:
            return 0.0

        std = float(np.std(r))
        return float(np.mean(r) / np.maximum(std, self.eps))

    # ---------- Z-score (runs test over win/loss sequence) ----------

    @property
    def z_score(self) -> float:

        """ Build win/loss sequence from CLOSED trades:"""

        seq = []
        for d in self.deals:
            if getattr(d, "entry", None) != MetaTrader5Constants.DEAL_ENTRY_OUT:
                continue
            seq.append(1 if float(getattr(d, "profit", 0.0)) > 0.0 else 0)

        n = len(seq)
        if n < 2:
            return 0.0

        n1 = sum(seq)  # wins
        n2 = n - n1  # losses
        if n1 == 0 or n2 == 0:
            return 0.0

        # number of runs
        R = 1
        for i in range(1, n):
            if seq[i] != seq[i - 1]:
                R += 1

        ER = 1.0 + (2.0 * n1 * n2) / (n1 + n2)
        VR = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1.0))
        if VR <= self.eps:
            return 0.0
        return float((R - ER) / np.sqrt(VR))

    # ---------- AHPR / GHPR ----------

    @property
    def ahpr_factor(self) -> float:
        """ AHPR = (1 + r1) * (1 + r2) * ... * (1 + rn)^(1/n) - 1, where r_i are per-trade returns in fraction (e.g., 0.01 = +1%) """
        r = np.asarray(self._trade_returns, dtype=np.float64)
        if r.size == 0:
            return 1.0
        return float(1.0 + np.mean(r))

    @property
    def ahpr_percent(self) -> float:
        """ AHPR in percent form. """
        return float((self.ahpr_factor - 1.0) * 100.0)

    @property
    def ghpr_factor(self) -> float:
        """ GHPR = ((1 + r1) * (1 + r2) * ... * (1 + rn))^(1/n) - 1, where r_i are per-trade returns in fraction (e.g., 0.01 = +1%) """
        r = np.asarray(self._trade_returns, dtype=np.float64)
        if r.size == 0:
            return 1.0
        return float(np.prod(1.0 + r) ** (1.0 / r.size))

    @property
    def ghpr_percent(self) -> float:
        """ GHPR in percent form. """
        return float((self.ghpr_factor - 1.0) * 100.0)

    @property
    def lr_correlation(self) -> float:
        """ Correlation coefficient (r-value) from linear regression of balance curve over time. """
        return np.nan if not self.lr_res else self.lr_res.rvalue

    @property
    def lr_standard_error(self) -> float:
        """ Standard error of the estimate from linear regression of balance curve over time. """
        return np.nan if not self.lr_res else self.lr_res.stderr

    @property
    def on_tester_results(self) -> float:
        return 0.0

    @property
    def margin_level(self) -> float:
        """ Minimum margin level (%) during the test. """
        return np.min(self.margin_level_curve) if len(self.margin_level_curve) > 0 else np.nan

    @staticmethod
    def holding_time_calculator(entry_time: pd.Series, exit_time: pd.Series) -> dict:
        """ Calculates holding time statistics (min, max, average) for trades based on entry and exit times of positions. """
        durations = exit_time - entry_time

        if durations.empty:
            return {"min": None, "max": None, "avg": None, "count": 0}

        return {
            "time": exit_time,
            "durations": durations,
            "count": int(len(durations)),
            "min": durations.min(),
            "max": durations.max(),
            "avg": durations.mean(),
        }

    def to_dict(self) -> dict:
        """
            Converts all non-protected methods in this class to a dictionary.

            Returns:
                   A dictionary object (dict).
        """
        result = {}

        for name, obj in inspect.getmembers(type(self)):

            # skip private/protected
            if name.startswith("_"):
                continue

            # collect @property values
            if isinstance(obj, property):
                try:
                    result[name] = getattr(self, name)
                except Exception:
                    result[name] = None

        return result

class EntriesCalculator:
    """ Calculates entry counts by hour, weekday, and month based on the deals data. """

    def __init__(self, deals_df: pd.DataFrame):
        self.deals_df = deals_df.query(
            f"entry=={MetaTrader5Constants.DEAL_ENTRY_IN} and (type=={MetaTrader5Constants.DEAL_TYPE_SELL} or type=={MetaTrader5Constants.DEAL_TYPE_BUY})").copy()

        time_col = self.deals_df["time"]

        # Convert only if not already datetime
        if not pd.api.types.is_datetime64_any_dtype(time_col):

            # If timestamps are seconds since epoch
            if pd.api.types.is_integer_dtype(time_col):
                self.deals_df["time"] = pd.to_datetime(
                    time_col,
                    unit="s",
                    errors="coerce"
                )

            else:
                self.deals_df["time"] = pd.to_datetime(
                    time_col,
                    errors="coerce"
                )

        self.deals_df["hour"] = self.deals_df["time"].dt.hour
        self.deals_df["weekday"] = self.deals_df["time"].dt.weekday
        self.deals_df["month"] = self.deals_df["time"].dt.month

    def by_hour(self) -> pd.Series:
        """ Returns a Series with the count of entries for each hour of the day (0-23). """
        return self.deals_df.groupby("hour")["entry"].size().reindex(range(24), fill_value=0)

    def by_weekday(self) -> pd.Series:
        """ Returns a Series with the count of entries for each weekday (0=Monday, 6=Sunday). """
        return self.deals_df.groupby("weekday")["entry"].size().reindex(range(7), fill_value=0)

    def by_month(self) -> pd.Series:
        """ Returns a Series with the count of entries for each month (1-12). """
        return self.deals_df.groupby("month")["entry"].size().reindex(range(1, 13), fill_value=0)


class PLCalculator:
    """ Calculates profit and loss statistics by hour, weekday, and month based on the deals data. """

    def __init__(self, deals_df: pd.DataFrame):
        """
        Args:
            deals_df (pd.DataFrame): DataFrame containing deal records with columns like 'entry', 'type', 'profit', 'commission', and 'time'.
        """

        self.deals_df = deals_df.query(
            f"entry == {MetaTrader5Constants.DEAL_ENTRY_OUT} and (type=={MetaTrader5Constants.DEAL_TYPE_BUY} | type=={MetaTrader5Constants.DEAL_TYPE_SELL})").copy()

        time_col = self.deals_df["time"]

        # Convert only if not already datetime
        if not pd.api.types.is_datetime64_any_dtype(time_col):

            # If timestamps are seconds since epoch
            if pd.api.types.is_integer_dtype(time_col):
                self.deals_df["time"] = pd.to_datetime(
                    time_col,
                    unit="s",
                    errors="coerce"
                )

            else:
                self.deals_df["time"] = pd.to_datetime(
                    time_col,
                    errors="coerce"
                )

        self.deals_df["hour"] = self.deals_df["time"].dt.hour
        self.deals_df["weekday"] = self.deals_df["time"].dt.weekday
        self.deals_df["month"] = self.deals_df["time"].dt.month

        net = (self.deals_df["profit"] + self.deals_df["commission"])

        self.deals_df["profit"] = net.clip(lower=0.0)
        self.deals_df["loss"] = net.clip(upper=0.0)

    def loss_by_hour(self) -> pd.Series:
        """ Returns a Series with the total loss for each hour of the day (0-23). """
        return self.deals_df.groupby("hour")["loss"].sum().reindex(range(24), fill_value=0)

    def profit_by_hour(self) -> pd.Series:
        """ Returns a Series with the total profit for each hour of the day (0-23). """
        return self.deals_df.groupby("hour")["profit"].sum().reindex(range(24), fill_value=0)

    def loss_by_weekday(self) -> pd.Series:
        """ Returns a Series with the total loss for each weekday (0=Monday, 6=Sunday). """

        return self.deals_df.groupby("weekday")["loss"].sum().reindex(range(7), fill_value=0)

    def profit_by_weekday(self) -> pd.Series:
        """ Returns a Series with the total profit for each weekday (0=Monday, 6=Sunday). """
        return self.deals_df.groupby("weekday")["profit"].sum().reindex(range(7), fill_value=0)

    def loss_by_month(self) -> pd.Series:
        """ Returns a Series with the total loss for each month (1-12). """
        return self.deals_df.groupby("month")["loss"].sum().reindex(range(1, 13), fill_value=0)

    def profit_by_month(self) -> pd.Series:
        """ Returns a Series with the total profit for each month (1-12). """
        return self.deals_df.groupby("month")["profit"].sum().reindex(range(1, 13), fill_value=0)