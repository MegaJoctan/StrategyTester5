from datetime import datetime
from strategytester5 import *

class SwapCalculator:
    """
    Simulates MetaTrader5 swap calculation for positions.
    """

    def __init__(self, mt5_instance, account_currency_getter, symbol_info_getter, tick_getter):
        """
        Parameters:
        - mt5_instance: reference to MT5 constants and enums
        - account_currency_getter: function that returns account currency
        - symbol_info_getter: function(symbol) -> symbol info object
        - tick_getter: function(symbol) -> latest tick
        """
        self.mt5 = mt5_instance
        self.get_account_currency = account_currency_getter
        self.get_symbol_info = symbol_info_getter
        self.get_tick = tick_getter

    # ---------------------- PUBLIC METHOD ----------------------
    
    def calc_swap(self, pos):
        """
        Calculate swap for a position for ONE day (triple swap handled separately)
        """
        symbol_info = self.get_symbol_info(pos.symbol)
        account_currency = self.get_account_currency()
        swap_mode = symbol_info.swap_mode
        swap_long = symbol_info.swap_long
        swap_short = symbol_info.swap_short
        contract_size = symbol_info.contract_size

        order_type = pos.type
        volume = pos.volume

        if swap_mode == self.mt5.SYMBOL_SWAP_MODE_DISABLED:
            return 0.0

        if swap_mode == self.mt5.SYMBOL_SWAP_MODE_POINTS:
            point_value = self._calc_point_value(pos.symbol, is_profit=(swap_long if order_type == self.mt5.POSITION_TYPE_BUY else swap_short) > 0)
            rate = swap_long if order_type == self.mt5.POSITION_TYPE_BUY else swap_short
            return volume * rate * point_value

        if swap_mode == self.mt5.SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT:
            rate = swap_long if order_type == self.mt5.POSITION_TYPE_BUY else swap_short
            return volume * rate

        if swap_mode in (self.mt5.SYMBOL_SWAP_MODE_CURRENCY_SYMBOL,
                         self.mt5.SYMBOL_SWAP_MODE_CURRENCY_MARGIN):
            rate = swap_long if order_type == self.mt5.POSITION_TYPE_BUY else swap_short
            swap = volume * rate

            if swap_mode == self.mt5.SYMBOL_SWAP_MODE_CURRENCY_SYMBOL:
                base_currency = symbol_info.currency_base
            else:
                base_currency = symbol_info.currency_margin

            if base_currency != account_currency:
                swap *= self._get_exchange_rate(base_currency, account_currency, order_type)
            return swap

        if swap_mode in (self.mt5.SYMBOL_SWAP_MODE_INTEREST_CURRENT,
                         self.mt5.SYMBOL_SWAP_MODE_INTEREST_OPEN):
            trade_value = self._trade_notional_value(pos, symbol_info, account_currency)
            rate = swap_long if order_type == self.mt5.POSITION_TYPE_BUY else swap_short
            return trade_value * rate / 100 / 360

        # Reopen modes (not implemented in tester)
        if swap_mode in (self.mt5.SYMBOL_SWAP_MODE_REOPEN_CURRENT,
                         self.mt5.SYMBOL_SWAP_MODE_REOPEN_BID):
            return 0.0

        return 0.0

    # ---------------------- INTERNAL HELPERS ----------------------
    def _calc_point_value(self, symbol, is_profit=True):
        """
        Approximate MT5 mPointValue
        """
        tick = self.get_tick(symbol)
        symbol_info = self.get_symbol_info(symbol)
        point = symbol_info.point
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value  # approximate
        return (point / tick_size) * tick_value

    def _trade_notional_value(self, pos, symbol_info, account_currency):
        """
        Equivalent to TradeNotionalValue() in MQL5
        """
        contract_size = symbol_info.contract_size
        profit_currency = symbol_info.currency_profit
        price = self.get_tick(pos.symbol).ask if pos.type == self.mt5.POSITION_TYPE_BUY else self.get_tick(pos.symbol).bid
        cross_rate = self._get_exchange_rate(profit_currency, account_currency, pos.type)
        return pos.volume * contract_size * price * cross_rate

    def _get_exchange_rate(self, ccy1, ccy2, order_type):
        """
        Recursive lookup for currency conversion, mirrors MQL5 logic
        """
        if ccy1 == ccy2:
            return 1.0

        direct = self._get_symbol_by_currencies(ccy1, ccy2)
        if direct:
            tick = self.get_tick(direct)
            return tick.bid if order_type == self.mt5.POSITION_TYPE_SELL else tick.ask

        reverse = self._get_symbol_by_currencies(ccy2, ccy1)
        if reverse:
            tick = self.get_tick(reverse)
            rate = tick.ask if order_type == self.mt5.POSITION_TYPE_SELL else tick.bid
            return 1.0 / rate

        # fallback via USD
        if ccy1 != "USD" and ccy2 != "USD":
            return self._get_exchange_rate(ccy1, "USD", order_type) * self._get_exchange_rate("USD", ccy2, order_type)

        return 1.0

    def _get_symbol_by_currencies(self, ccy1, ccy2):
        """
        Implement symbol lookup logic in tester (or return None if not found)
        """
        # Example: return f"{ccy1}{ccy2}" if in self.symbols else None
        return None
