import numpy as np
from numba import njit
from strategytester5 import MetaTrader5
from scipy.stats import linregress

@njit(cache=True)
def _maximal_drawdown_local_extrema_nb(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    MT5-style maximal drawdown:
      max over local maxima (peak - next local minimum after that peak)

    Numba-optimized:
      - preallocates peaks/troughs arrays
      - uses counters instead of Python lists
      - plateau handling (flat runs)
    """
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

            d_type = getattr(d, "type", None)
            if d_type == MetaTrader5.DEAL_TYPE_BUY:
                self._total_long_trades += 1
            elif d_type == MetaTrader5.DEAL_TYPE_SELL:
                self._total_short_trades += 1

            profit = float(getattr(d, "profit", 0.0))

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
        return np.sum(np.abs(self._losses)) if self._losses else 0.0

    @property
    def net_profit(self) -> float:
        return self.gross_profit - self.gross_loss

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / max(self.gross_loss, 0.1)

    @property
    def recovery_factor(self) -> float:
        return self.net_profit / max(self.balance_drawdown_maximal, self.eps)

    @property
    def expected_payoff(self) -> int:
        return (self.net_profit / self.total_trades) if self.total_trades > 0 else 0

    @property
    def equity_drawdown_absolute(self) -> float:
        return self.initial_deposit - np.min(self.equity_curve)

    @property
    def equity_drawdown_absolute_percent(self) -> float:
        return 0.0

    @property
    def equity_drawdown_relative(self) -> float:
        return self.equity_drawdown_absolute / (np.max(self.equity_curve) + self.eps) * 100

    @property
    def balance_drawdown_absolute(self) -> float:
        return self.initial_deposit - np.min(self.balance_curve)

    @property
    def balance_drawdown_absolute_percent(self) -> float:
        return 0.0

    @property
    def balance_drawdown_relative(self) -> float:
        return self.balance_drawdown_absolute / (np.max(self.balance_curve) + self.eps) * 100

    def _maximal_drawdown(self, curve) -> float:
        """
        MT5-style maximal drawdown (wrapper around a Numba kernel).
        """
        arr = np.ascontiguousarray(np.asarray(curve, dtype=np.float64))
        return float(_maximal_drawdown_local_extrema_nb(arr))

    @property
    def balance_drawdown_maximal(self) -> float:
        return self._maximal_drawdown(self.balance_curve)

    @property
    def equity_drawdown_maximal(self) -> float:
        return self._maximal_drawdown(self.equity_curve)

    @property
    def sharpe_ratio(self) -> float:
        std = np.std(self._returns)
        return float(np.mean(self._returns) / np.maximum(std, self.eps))

    # ---------- Z-score (runs test over win/loss sequence) ----------

    @property
    def z_score(self) -> float:
        # MT5 series test (runs). :contentReference[oaicite:19]{index=19}
        # Build win/loss sequence from CLOSED trades:
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

    @property
    def ahpr(self) -> float:
        return np.prod(1 + self._returns) ** (1 / len(self._returns)) if len(self._returns) else 0

    @property
    def ghpr(self) -> float:
        return np.prod(1 + self._returns) if len(self._returns) else 0

    @property
    def lr_correlation(self) -> float:
        return self.lr_res.rvalue

    @property
    def lr_standard_error(self) -> float:
        return self.lr_res.stderr

    @property
    def on_tester_results(self) -> float:
        return 0.0

    @property
    def margin_level(self) -> float:
        return np.min(self.margin_level_curve)
