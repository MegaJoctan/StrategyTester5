import numpy as np
from numba import njit

@njit
def _resolve_tick_count(tv: int) -> int:
    if tv < 1:
        return 1
    if tv > 20:
        return 20
    return tv

@njit
def _support_points(o, h, l, c):
    # returns 4 points in fixed array
    pts = np.empty(4, dtype=np.float64)
    if c >= o:  # bullish
        pts[0] = o; pts[1] = l; pts[2] = h; pts[3] = c
    else:       # bearish
        pts[0] = o; pts[1] = h; pts[2] = l; pts[3] = c
    return pts

@njit
def get_ticks_from_bars(
    opens, highs, lows, closes,
    spreads, tick_volumes,
    times_msc,
    symbol_point: float
):
    """
    Returns (bid, ask, time_msc, out_count)
    Strategy:
      - Preallocate worst-case N = len(bars)*20
      - Fill sequentially
    """
    n = opens.shape[0]
    max_ticks = n * 20

    out_bid  = np.empty(max_ticks, dtype=np.float64)
    out_ask  = np.empty(max_ticks, dtype=np.float64)
    out_tmsc = np.empty(max_ticks, dtype=np.int64)

    out_i = 0

    for j in range(n):
        o = opens[j]; h = highs[j]; l = lows[j]; c = closes[j]
        spr = spreads[j]
        tv = tick_volumes[j]
        base = times_msc[j]

        tick_count = _resolve_tick_count(tv)
        step = 1000 // tick_count
        if step < 1:
            step = 1

        # spread in price units
        spr_price = spr * symbol_point

        if tick_count == 1:
            price = c
            out_bid[out_i]  = price
            out_ask[out_i]  = price + spr_price
            out_tmsc[out_i] = base
            out_i += 1
            continue

        if tick_count == 2:
            out_bid[out_i]  = o
            out_ask[out_i]  = o + spr_price
            out_tmsc[out_i] = base
            out_i += 1

            out_bid[out_i]  = c
            out_ask[out_i]  = c + spr_price
            out_tmsc[out_i] = base + step
            out_i += 1
            continue

        pts = _support_points(o, h, l, c)  # 4 points
        segments = 3

        ticks_per_seg = tick_count // segments
        rem = tick_count % segments

        t_index = 0

        for seg in range(segments):
            start = pts[seg]
            end   = pts[seg + 1]
            steps = ticks_per_seg + (1 if seg < rem else 0)

            # edge-case guard
            if steps < 1:
                steps = 1

            if steps == 1:
                # single tick at end
                price = end
                out_bid[out_i]  = price
                out_ask[out_i]  = price + spr_price
                out_tmsc[out_i] = base + t_index * step
                out_i += 1
                t_index += 1
            else:
                # linear interpolation including endpoints
                denom = steps - 1
                for k in range(steps):
                    price = start + (end - start) * (k / denom)
                    out_bid[out_i]  = price
                    out_ask[out_i]  = price + spr_price
                    out_tmsc[out_i] = base + t_index * step
                    out_i += 1
                    t_index += 1

        # enforce cap (in case of rounding issues)
        # (Normally exact, but safe)
        # out_i already points to end; nothing to trim here because we don't overfill per bar.

    return out_bid[:out_i], out_ask[:out_i], out_tmsc[:out_i], out_i
