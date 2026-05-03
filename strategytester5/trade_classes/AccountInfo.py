from __future__ import annotations
from typing import Optional, Union
from collections import namedtuple

class CAccountInfo:
    """
        A lightweight Python wrapper that resembles the MQL5 Standard Library class
        `CAccountInfo` and provides convenient, read-only access to the properties of
        the currently connected MetaTrader 5 trading account.

        This class caches the result of `mt5.account_info()` at construction time.
        The returned values reflect the account state at the time of initialization
        (balance, equity, margin, etc.). If you need up-to-date values after trading
        activity or price changes, re-instantiate the class or implement a refresh.

        [Reference (MQL5)](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo)
    """

    def __init__(self, ac_info: namedtuple):
        
        """
        Instatiates the CAccountInfo object

        Args:
            ac_info : A tuple returned by `mt5.account_info()`

        Raises:
            RuntimeError : If account information cannot be retrieved (i.e., `account_info()` returns None).

        Notes:
            Method groups mirror the MQL5 layout:
            - Integer properties: Login, TradeMode, Leverage, StopoutMode, MarginMode, etc.
            - Double properties: Balance, Equity, Margin, FreeMargin, etc.
            - String properties: Name, Server, Currency, Company
            - Checks / calculations: MarginCheck, OrderProfitCheck, FreeMarginCheck, MaxLotCheck
        """

        if ac_info is None:
            raise RuntimeError("Failed to retrieve account info: ", self.terminal.last_error())

        self._account_info = ac_info

    # --- Integer properties
    
    @property
    def login(self) -> int:
        """Gets the account number (Login)."""
        
        return int(self._account_info.login)

    @property
    def trade_mode(self) -> int:
        """Gets the trade mode (ACCOUNT_TRADE_MODE_*)."""
        
        return int(self._account_info.trade_mode)

    @property
    def trade_mode_description(self) -> str:
        """Gets the trade mode as a human-readable string."""
        
        mode_map = {
            self.terminal.ACCOUNT_TRADE_MODE_DEMO: "Demo",
            self.terminal.ACCOUNT_TRADE_MODE_CONTEST: "Contest",
            self.terminal.ACCOUNT_TRADE_MODE_REAL: "Real",
        }
        
        return mode_map.get(self._account_info.trade_mode, "Unknown")

    @property
    def leverage(self) -> int:
        """Gets the account leverage."""
        
        return int(self._account_info.leverage)

    @property
    def stopout_mode(self) -> int:
        """Gets the stop-out mode (ACCOUNT_STOPOUT_MODE_*)."""
        
        return int(self._account_info.margin_so_mode)

    @property
    def stopout_mode_description(self) -> str:
        """Gets the stop-out mode as a human-readable string."""
        
        mode_map = {
            self.terminal.ACCOUNT_STOPOUT_MODE_PERCENT: "Percent",
            self.terminal.ACCOUNT_STOPOUT_MODE_MONEY: "Money",
        }
        
        return mode_map.get(self._account_info.margin_so_mode, "Unknown")

    @property
    def margin_mode(self) -> int:
        """Gets the margin calculation mode (ACCOUNT_MARGIN_MODE_*)."""
        
        return int(self._account_info.margin_mode)

    @property
    def margin_mode_description(self) -> str:
        """Gets the margin calculation mode as a human-readable string."""
        
        mode_map = {
            self.terminal.ACCOUNT_MARGIN_MODE_RETAIL_NETTING: "Retail Netting",
            self.terminal.ACCOUNT_MARGIN_MODE_EXCHANGE: "Exchange",
            self.terminal.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING: "Retail Hedging",
        }
        return mode_map.get(self._account_info.margin_mode, "Unknown")

    @property
    def trade_allowed(self) -> bool:
        """Returns True if trading is allowed for the account."""
        return bool(self._account_info.trade_allowed)

    @property
    def trade_expert(self) -> bool:
        """Returns True if automated trading is allowed for the account."""
        return bool(self._account_info.trade_expert)

    @property
    def limit_orders(self) -> int:
        """Gets the maximum number of allowed pending orders."""
        return int(self._account_info.limit_orders)

    # --- Double properties

    @property
    def balance(self) -> float:
        """Gets the account balance."""
        return float(self._account_info.balance)

    @property
    def credit(self) -> float:
        """Gets the account credit."""
        return float(self._account_info.credit)

    @property
    def profit(self) -> float:
        """Gets the current profit for the account."""
        return float(self._account_info.profit)

    @property
    def equity(self) -> float:
        """Gets the account equity."""
        return float(self._account_info.equity)

    @property
    def margin(self) -> float:
        """Gets the reserved margin."""
        return float(self._account_info.margin)

    @property
    def free_margin(self) -> float:
        """Gets the free margin."""
        return float(self._account_info.margin_free)

    @property
    def margin_level(self) -> float:
        """Gets the margin level."""
        return float(self._account_info.margin_level)

    @property
    def margin_call(self) -> float:
        """Gets the margin call level."""
        return float(self._account_info.margin_so_call)

    @property
    def margin_stopout(self) -> float:
        """Gets the stop-out level."""
        return float(self._account_info.margin_so_so)

    # --- String properties

    @property
    def name(self) -> str:
        """Gets the client name."""
        return str(self._account_info.name)

    @property
    def server(self) -> str:
        """Gets the trade server name."""
        return str(self._account_info.server)

    @property
    def currency(self) -> str:
        """Gets the deposit currency."""
        return str(self._account_info.currency)

    @property
    def company(self) -> str:
        """Gets the broker/company name."""
        return str(self._account_info.company)

    # --- Checks

    def order_profit_check(self, symbol: str, order_type: int, volume: float, price_open: float, price_close: float) -> float:
        
        """Evaluates profit for the given open/close prices (approximate, ignores commissions/swaps)."""
        
        profit = self.terminal.order_calc_profit(order_type, symbol, volume, price_open, price_close)
        return float(profit) if profit is not None else 0.0

    def free_margin_check(self, symbol: str, order_type: int, volume: float, price: float) -> Optional[float]:
        
        """Returns free margin left after opening the position, or None if margin check fails."""
        
        required_margin = self.margin_check(symbol, order_type, volume, price)
        return None if required_margin is None else (self.free_margin - required_margin)

    def margin_check(self, symbol: str, order_type: int, volume: float, price: float) -> Optional[float]:
        
        """Gets the amount of margin required to execute trade operation"""
        
        margin = self.terminal.order_calc_margin(order_type, symbol, volume, price)
        return margin if margin is not None else 0.0

    def max_lot_check(self, symbol: str, order_type: int, price: float, percent: float = 100.0) -> Optional[float]: 
        """ Estimates the maximum tradable volume based on available margin. 
        
        Args:
            percent : float Percentage of free margin to use (0..100).
        
        Returns:
            Estimated maximum lot size, or None if margin cannot be estimated. """
        
        required_margin_per_lot = self.margin_check(symbol, order_type, 1.0, price)
        if required_margin_per_lot in (None, 0.0): 
            return None 
        
        margin_available = self.free_margin * (percent / 100.0)
        return margin_available / required_margin_per_lot
    
    
    def print_all(self) -> None:
        """Print all @property values of the class."""
        
        for name, attr in vars(type(self)).items():
            if isinstance(attr, property):
                try:
                    value = getattr(self, name)
                except Exception as e:
                    value = f"<error: {e}>"
                
                print(f"{name:20} : {value}")