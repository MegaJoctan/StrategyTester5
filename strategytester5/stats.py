import numpy as np
from numba import njit
from strategytester5 import MetaTrader5
from scipy.stats import linregress
"""
@njit(cache=True)
def _maximal_drawdown_local_extrema_nb(x: np.ndarray, eps: float = 1e-12) -> float:
    \"""
    MT5-style maximal drawdown:
      max over local maxima (peak - next local minimum after that peak)

    Numba-optimized:
      - preallocates peaks/troughs arrays
      - uses counters instead of Python lists
      - plateau handling (flat runs)
    \"""
    n = x.size
    if n < 3:
        return 0.0

    # Worst-case: every point is an extremum
    peaks = np.empty(n, dtype=np.int64)
    troughs = np.empty(n, dtype=np.int64)
    pcount = 0
    tcount = 0

    i = 1
    while i < n - 1:
        left = x[i - 1]
        mid = x[i]

        # Expand plateau: [i .. j]
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
        return 0.0

    # Compute max drawdown: for each peak, find next trough after it
    max_dd = 0.0
    t_idx = 0

    for pi in range(pcount):
        p = peaks[pi]

        # Advance trough pointer to first trough after this peak
        while t_idx < tcount and troughs[t_idx] <= p:
            t_idx += 1

        if t_idx >= tcount:
            break

        t = troughs[t_idx]
        dd = x[p] - x[t]
        if dd > max_dd:
            max_dd = dd

    return max_dd if max_dd > 0.0 else 0.0
"""

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
    def __init__(self,
                deals: list,
                initial_deposit: float,
                balance_curve: np.ndarray,
                equity_curve: np.ndarray,
                margin_level_curve: np.ndarray,
                ticks: int,
                symbols: int
                ):

        self.deals = deals
        self.initial_deposit = float(initial_deposit)
        self.balance_curve = np.ascontiguousarray(np.asarray(balance_curve, dtype=np.float64)).reshape(-1)
        self.equity_curve = np.ascontiguousarray(np.asarray(equity_curve, dtype=np.float64)).reshape(-1)
        self.margin_level_curve = np.ascontiguousarray(np.asarray(margin_level_curve, dtype=np.float64)).reshape(-1)
        self.ticks = ticks
        self.symbols = symbols

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
            if getattr(d, "entry", None) != MetaTrader5.DEAL_ENTRY_OUT:
                continue

            self._total_trades += 1

            d_type = d.type
            if d_type == MetaTrader5.DEAL_TYPE_BUY:
                self._total_long_trades += 1
            elif d_type == MetaTrader5.DEAL_TYPE_SELL:
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

                if d_type == MetaTrader5.DEAL_TYPE_BUY:
                    self._long_trades_won += 1
                elif d_type == MetaTrader5.DEAL_TYPE_SELL:
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
    def total_trades(self) -> int:
        return self._total_trades

    @property
    def total_deals(self) -> int:
        return len(self.deals)-1

    @property
    def total_short_trades(self) -> int:
        return self._total_short_trades

    @property
    def total_long_trades(self) -> int:
        return self._total_long_trades

    @property
    def short_trades_won(self) -> int:
        return self._short_trades_won

    @property
    def long_trades_won(self) -> int:
        return self._long_trades_won

    @property
    def profit_trades(self) -> int:
        return len(self._profits) if self._profits else 0

    @property
    def loss_trades(self) -> int:
        return len(self._losses) if self._losses else 0

    @property
    def largest_profit_trade(self) -> float:
        return np.max(self._profits) if self._profits else 0

    @property
    def largest_loss_trade(self) -> float:
        return np.min(self._losses) if self._losses else 0

    @property
    def average_profit_trade(self) -> float:
        return np.mean(self._profits) if self._profits else 0

    @property
    def average_loss_trade(self) -> float:
        return np.mean(self._losses) if self._losses else 0

    # ---------- streak metrics ----------

    @property
    def maximum_consecutive_wins_count(self) -> int:
        return self._max_consec_win_count

    @property
    def maximum_consecutive_wins_money(self) -> float:
        return self._max_consec_win_money

    @property
    def maximum_consecutive_losses_count(self) -> int:
        return self._max_consec_loss_count

    @property
    def maximum_consecutive_losses_money(self) -> float:
        # show as absolute money if you prefer; MT5 shows total loss (negative) in brackets
        return self._max_consec_loss_money

    @property
    def maximal_consecutive_profit_count(self) -> int:
        return self._max_profit_streak_count

    @property
    def maximal_consecutive_profit_money(self) -> float:
        return self._max_profit_streak_money

    @property
    def maximal_consecutive_loss_count(self) -> int:
        return self._max_loss_streak_count

    @property
    def maximal_consecutive_loss_money(self) -> float:
        return self._max_loss_streak_money

    @property
    def average_consecutive_wins(self) -> float:
        return np.mean(self._win_streaks) if self._win_streaks else 0

    @property
    def average_consecutive_losses(self) -> float:
        return np.mean(self._loss_streaks) if self._loss_streaks else 0

    @property
    def gross_profit(self) -> float:
        return np.sum(self._profits) if self._profits else 0.0

    @property
    def gross_loss(self) -> float:
        return np.sum(self._losses) if self._losses else 0.0

    @property
    def net_profit(self) -> float:
        return self.gross_profit - np.abs(self.gross_loss)

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / (self.gross_loss + self.eps)

    @property
    def recovery_factor(self) -> float:
        return self.net_profit / (self.equity_drawdown_maximal + self.eps)

    @property
    def expected_payoff(self) -> int:
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
        # AbsoluteDrawDown = InitialDeposit - MinimalBalance (below initial) :contentReference[oaicite:12]{index=12}
        return self._abs_drawdown(self.initial_deposit, self.balance_curve)

    @property
    def equity_drawdown_absolute(self) -> float:
        return self._abs_drawdown(self.initial_deposit, self.equity_curve)

    @property
    def balance_drawdown_maximal(self) -> float:
        return float(_max_dd_money_and_pct_nb(self.balance_curve)[0])

    def _validate_baleq_values(self, value: float) -> float:
        if abs(value) > self.initial_deposit:
            return np.nan

        return value

    @property
    def balance_drawdown_relative(self) -> float:
        """ MT5 relative drawdown uses local high -> next local low (max %)"""
        return self._validate_baleq_values(float(_max_dd_money_and_pct_nb(self.balance_curve)[1]))

    @property
    def equity_drawdown_maximal(self) -> float:
        return self._validate_baleq_values(float(_max_dd_money_and_pct_nb(self.equity_curve)[0]))

    @property
    def equity_drawdown_relative(self) -> float:
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
            if getattr(d, "entry", None) != MetaTrader5.DEAL_ENTRY_OUT:
                continue
            seq.append(1 if float(getattr(d, "profit", 0.0)) > 0.0 else 0)

        n = len(seq)
        if n < 2:
            return 0.0

        n1 = sum(seq)          # wins
        n2 = n - n1            # losses
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

    """
    @property
    def ahpr(self) -> float:

        \""" arithmetic mean of trade returns (fraction). In report they also show % in brackets.\"""

        r = np.asarray(self._trade_returns, dtype=np.float64)
        return float(np.mean(r)) if r.size else 0.0

    @property
    def ghpr(self) -> float:
        \""" geometric mean of trade returns (fraction)\"""

        r = np.asarray(self._trade_returns, dtype=np.float64)
        if r.size == 0:
            return 0.0
        g = float(np.prod(1.0 + r) ** (1.0 / r.size) - 1.0)
        return g
    """

    @property
    def ahpr_factor(self) -> float:
        r = np.asarray(self._trade_returns, dtype=np.float64)
        if r.size == 0:
            return 1.0
        return float(1.0 + np.mean(r))

    @property
    def ahpr_percent(self) -> float:
        return float((self.ahpr_factor - 1.0) * 100.0)

    @property
    def ghpr_factor(self) -> float:
        r = np.asarray(self._trade_returns, dtype=np.float64)
        if r.size == 0:
            return 1.0
        return float(np.prod(1.0 + r) ** (1.0 / r.size))

    @property
    def ghpr_percent(self) -> float:
        return float((self.ghpr_factor - 1.0) * 100.0)

    @property
    def lr_correlation(self) -> float:
        return np.nan if not self.lr_res else self.lr_res.rvalue

    @property
    def lr_standard_error(self) -> float:
        return np.nan if not self.lr_res else self.lr_res.stderr

    @property
    def on_tester_results(self) -> float:
        return 0.0

    @property
    def margin_level(self) -> float:
        return np.min(self.margin_level_curve) if len(self.margin_level_curve)>0 else np.nan
