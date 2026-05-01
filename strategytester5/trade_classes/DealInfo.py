from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Union
from ..MetaTrader5.constants import MetaTrader5Constants as mt5_constants
from .. import TradeDeal

class CDealInfo:
    """
        A lightweight Python wrapper that resembles the MQL5 Standard Library class
        `CDealInfo` and provides convenient, read-only access to MetaTrader 5 deal
        properties.

        [Reference (MQL5)](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo)
    """

    def __init__(self, deal: TradeDeal):
        """
        Instantiates the CDealInfo object

        Args:
            deal: A deal object (e.g., from `mt5.history_deals_get()`) to wrap and provide access to.

        Notes:
            - Many getters return `None` if no deal is selected.
            - Times:
                * `time` returns a timezone-aware UTC datetime (recommended for MT5 history).
                * `time_msc` returns milliseconds since 1970-01-01 (UTC) when available.
            - This wrapper does not modify terminal state; it only reads/derives properties.
        """

        if self.deal is None:
            raise ValueError("Deal cannot be None. Please provide a valid deal tuple.")

        self._deal = deal

    @property
    def deal_type_description(self):
        """Gets the deal type as a string"""

        if not self._deal:
            return "N/A"
        
        deal_type_map = {
            mt5_constants.DEAL_TYPE_BUY: "BUY",
            mt5_constants.DEAL_TYPE_SELL: "SELL",
            mt5_constants.DEAL_TYPE_BALANCE: "BALANCE",
            mt5_constants.DEAL_TYPE_CREDIT: "CREDIT",
            mt5_constants.DEAL_TYPE_CHARGE: "CHARGE",
            mt5_constants.DEAL_TYPE_CORRECTION: "CORRECTION",
            mt5_constants.DEAL_TYPE_BONUS: "BONUS",
            mt5_constants.DEAL_TYPE_COMMISSION: "COMMISSION",
            mt5_constants.DEAL_TYPE_COMMISSION_DAILY: "COMMISSION DAILY",
            mt5_constants.DEAL_TYPE_COMMISSION_MONTHLY: "COMMISSION MONTHLY",
            mt5_constants.DEAL_TYPE_COMMISSION_AGENT_DAILY: "AGENT COMMISSION DAILY",
            mt5_constants.DEAL_TYPE_COMMISSION_AGENT_MONTHLY: "AGENT COMMISSION MONTHLY",
            mt5_constants.DEAL_TYPE_INTEREST: "INTEREST",
            mt5_constants.DEAL_TYPE_BUY_CANCELED: "BUY CANCELED",
            mt5_constants.DEAL_TYPE_SELL_CANCELED: "SELL CANCELED"
        }
        
        return deal_type_map.get(self._deal.type, f"UNKNOWN({self._deal.type})")

    @property
    def entry_description(self):
        """Gets the deal direction as a string"""

        if not self._deal:
            return "N/A"

        entry_map = {
            mt5_constants.DEAL_ENTRY_IN: "IN",
            mt5_constants.DEAL_ENTRY_OUT: "OUT",
            mt5_constants.DEAL_ENTRY_INOUT: "INOUT"
        }

        return entry_map.get(self._deal.entry, "UNKNOWN")

    # --- Integer, datetime & properties

    @property
    def ticket(self) -> Optional[int]:
        """Gets ticket/selects the deal ticket."""
        return None if self._deal is None else int(self._deal.ticket)

    @property
    def order(self) -> Optional[int]:
        """Gets the order by which the deal was executed."""
        return None if self._deal is None else int(self._deal.order)

    @property
    def time(self) -> Optional[datetime]:
        """
        Gets the time of deal execution as a timezone-aware UTC datetime.

        Returns:
            UTC datetime, or None if no deal is selected.
        """
        if self._deal is None:
            return None
        # MT5 history times are effectively UTC; keep it explicit.
        return datetime.fromtimestamp(int(self._deal.time), tz=timezone.utc)

    @property
    def time_msc(self) -> Optional[int]:
        """Receives the time of a deal execution in milliseconds since 1970-01-01 (UTC)."""
        return None if self._deal is None else int(self._deal.time_msc)

    @property
    def deal_type(self) -> Optional[int]:
        """Gets the deal type (DEAL_TYPE_*)."""
        return None if self._deal is None else int(self._deal.type)

    @property
    def type_description(self) -> str:
        """Gets the deal type as a string."""
        
        if self._deal is None:
            return "N/A"
        
        deal_type_map ={
            mt5_constants.DEAL_TYPE_BUY: "BUY",
            mt5_constants.DEAL_TYPE_SELL: "SELL",
            mt5_constants.DEAL_TYPE_BALANCE: "BALANCE",
            mt5_constants.DEAL_TYPE_CREDIT: "CREDIT",
            mt5_constants.DEAL_TYPE_CHARGE: "CHARGE",
            mt5_constants.DEAL_TYPE_CORRECTION: "CORRECTION",
            mt5_constants.DEAL_TYPE_BONUS: "BONUS",
            mt5_constants.DEAL_TYPE_COMMISSION: "COMMISSION",
            mt5_constants.DEAL_TYPE_COMMISSION_DAILY: "COMMISSION DAILY",
            mt5_constants.DEAL_TYPE_COMMISSION_MONTHLY: "COMMISSION MONTHLY",
            mt5_constants.DEAL_TYPE_COMMISSION_AGENT_DAILY: "AGENT COMMISSION DAILY",
            mt5_constants.DEAL_TYPE_COMMISSION_AGENT_MONTHLY: "AGENT COMMISSION MONTHLY",
            mt5_constants.DEAL_TYPE_INTEREST: "INTEREST",
            mt5_constants.DEAL_TYPE_BUY_CANCELED: "BUY CANCELED",
            mt5_constants.DEAL_TYPE_SELL_CANCELED: "SELL CANCELED",
        }

        return deal_type_map.get(self._deal.type, f"UNKNOWN({self._deal.type})")

    @property
    def entry(self) -> Optional[int]:
        """Gets the deal direction/entry (DEAL_ENTRY_*)."""
        return None if self._deal is None else int(self._deal.entry)

    @property
    def magic(self) -> Optional[int]:
        """Gets the ID of the expert (magic number) that executed the deal."""
        return None if self._deal is None else int(self._deal.magic)

    @property
    def position_id(self) -> Optional[int]:
        """Gets the ID of the position in which the deal was involved."""
        return None if self._deal is None else int(self._deal.position_id)

    # Double properties

    @property
    def volume(self) -> Optional[float]:
        """Gets the volume of the deal."""
        return None if self._deal is None else float(self._deal.volume)

    @property
    def price(self) -> Optional[float]:
        """Gets the deal price."""
        return None if self._deal is None else float(self._deal.price)

    @property
    def commission(self) -> Optional[float]:
        """Gets the amount of commission for the deal."""
        return None if self._deal is None else float(self._deal.commission)

    @property
    def swap(self) -> Optional[float]:
        """Gets the amount of swap when the position is closed."""
        return None if self._deal is None else float(self._deal.swap)

    @property
    def profit(self) -> Optional[float]:
        """Gets the financial result of the deal."""
        return None if self._deal is None else float(self._deal.profit)

    # Text properties

    @property
    def symbol(self) -> Optional[str]:
        """Gets the symbol name."""
        return None if self._deal is None else str(self._deal.symbol)

    @property
    def comment(self) -> Optional[str]:
        """Gets the deal comment."""
        return None if self._deal is None else str(self._deal.comment)

    @property
    def external_id(self) -> Optional[str]:
        """Gets the external ID (if provided by the broker)."""
        return None if self._deal is None else str(self._deal.external_id)

    # -------------------------
    # Generic "Info*" methods (MQL5-like)
    # -------------------------

    def info_integer(self, prop_name: str) -> Optional[int]:
        """
        Gets an integer property value by attribute name.

        This is a Pythonic approximation of MQL5's InfoInteger/HistoryDealGetInteger.
        In Python MT5, deal objects expose fields as attributes.

        Args:
            prop_name: Attribute name on the deal object (e.g., "ticket", "order", "magic").

        Returns:
             The integer value, or None if no deal is selected or attribute missing.
        """

        if self._deal is None or not hasattr(self._deal, prop_name):
            return None
        value = getattr(self._deal, prop_name)
        return None if value is None else int(value)

    def info_double(self, prop_name: str) -> Optional[float]:
        """
        Gets a float property value by attribute name.

        Args:
            prop_name : Attribute name on the deal object (e.g., "price", "profit", "volume").

        Returns:
            The float value, or None if no deal is selected or attribute missing.
        """
        if self._deal is None or not hasattr(self._deal, prop_name):
            return None
        value = getattr(self._deal, prop_name)
        return None if value is None else float(value)

    def info_string(self, prop_name: str) -> Optional[str]:
        """
        Gets a string property value by attribute name.

        Args:
            prop_name: Attribute name on the deal object (e.g., "symbol", "comment", "external_id").

        Returns:
            The string value, or None if no deal is selected or attribute missing.
        """
        if self._deal is None or not hasattr(self._deal, prop_name):
            return None
        value = getattr(self._deal, prop_name)
        return None if value is None else str(value)
    
    def print_all(self) -> None:
        """Print all @property values of the class."""
        
        for name, attr in vars(type(self)).items():
            if isinstance(attr, property):
                try:
                    value = getattr(self, name)
                except Exception as e:
                    value = f"<error: {e}>"
                
                print(f"{name:20} : {value}")