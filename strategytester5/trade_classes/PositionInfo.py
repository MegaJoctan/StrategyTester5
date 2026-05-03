from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Union
from ..MetaTrader5 import TradePosition

class CPositionInfo:
    """
        CPositionInfo
        -------------
        A lightweight Python wrapper that resembles the MQL5 Standard Library class
        `CPositionInfo` and provides convenient, read-only access to MetaTrader 5
        open position properties.

        This class wraps a single already-selected/open position object returned by
        the MetaTrader 5 Python API, such as from `mt5.positions_get()`.

        [Reference (MQL5)](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo)
    """

    def __init__(self, position: TradePosition) -> None:
        
        """
        Instantiates a CPositionInfo object.

        Parameters
        ----------
        position : Any
            A position object returned by MetaTrader 5 Python API functions such as
            `mt5.positions_get()`.

        Notes
        -----
        - This wrapper does not implement selection methods by design.
        - If no position is supplied, properties return fallback values such as
        `-1`, `0.0`, `""`, or `None`, depending on the property.
        - Time values are returned as timezone-aware UTC datetimes where applicable.

        Method groups mirror the MQL5 layout:
        - Integer properties: Time, TimeMsc, TimeUpdate, TimeUpdateMsc, PositionType, Magic, Identifier
        - Double properties: Volume, PriceOpen, StopLoss, TakeProfit, PriceCurrent, Swap, Profit
        - String properties: Symbol, Comment
        - Generic accessors: InfoInteger, InfoDouble, InfoString
        """

        self._position: Optional[Any] = position
        
        if position is None or not hasattr(position, "ticket"):
            raise ValueError("The provided position object does not have expected attributes.")

    # --------- Integer / datetime properties -----------

    @property
    def ticket(self) -> int:
        """Gets the ticket / identifier of the position."""
        return int(self._position.ticket) if self._position else -1

    @property
    def identifier(self) -> int:
        """Gets the ID of the position."""
        return int(self._position.identifier) if self._position else -1

    @property
    def magic(self) -> int:
        """Gets the ID of the expert that opened the position."""
        return int(self._position.magic) if self._position else -1

    @property
    def time(self) -> Optional[datetime]:
        """Gets the time of position opening."""
        return datetime.fromtimestamp(self._position.time, tz=timezone.utc) if self._position else None

    @property
    def time_msc(self) -> Optional[int]:
        """Receives the time of position opening in milliseconds since 01.01.1970."""
        return int(self._position.time_msc) if self._position else None

    @property
    def time_update(self) -> Optional[datetime]:
        """Receives the time of position changing in seconds since 01.01.1970."""
        return datetime.fromtimestamp(self._position.time_update, tz=timezone.utc) if self._position else None

    @property
    def time_update_msc(self) -> Optional[int]:
        """Receives the time of position changing in milliseconds since 01.01.1970."""
        return int(self._position.time_update_msc) if self._position else None

    @property
    def position_type(self) -> int:
        """Gets the position type."""
        return int(self._position.type) if self._position else -1

    @property
    def position_type_description(self) -> str:
        """Gets the position type as a string."""
        pos_type_map = {
            self.terminal.POSITION_TYPE_BUY: "Buy",
            self.terminal.POSITION_TYPE_SELL: "Sell"
        }
        return pos_type_map.get(self.position_type, "Unknown position type")

    # ----------- Double properties -------------

    @property
    def volume(self) -> float:
        """Gets the volume of the position."""
        return float(self._position.volume) if self._position else 0.0

    @property
    def price_open(self) -> float:
        """Gets the opening price of the position."""
        return float(self._position.price_open) if self._position else 0.0

    @property
    def stop_loss(self) -> float:
        """Gets the Stop Loss price of the position."""
        return float(self._position.sl) if self._position else 0.0

    @property
    def take_profit(self) -> float:
        """Gets the Take Profit price of the position."""
        return float(self._position.tp) if self._position else 0.0

    @property
    def price_current(self) -> float:
        """Gets the current price by position symbol."""
        return float(self._position.price_current) if self._position else 0.0

    @property
    def swap(self) -> float:
        """Gets the swap amount of the position."""
        return float(self._position.swap) if self._position else 0.0

    @property
    def profit(self) -> float:
        """Gets the current profit of the position."""
        return float(self._position.profit) if self._position else 0.0

    # Optional: some MT5 Python position objects may not expose commission
    @property
    def commission(self) -> float:
        """Gets the commission amount of the position, if available."""
        if not self._position:
            return 0.0
        value = getattr(self._position, "commission", 0.0)
        return float(value) if value is not None else 0.0

    # ------------ String properties ------------

    @property
    def symbol(self) -> str:
        """Gets the position symbol."""
        return str(self._position.symbol) if self._position else ""

    @property
    def comment(self) -> str:
        """Gets the comment of the position."""
        return str(self._position.comment) if self._position else ""

    # ------------ Generic Info* methods ------------

    def info_integer(self, prop_name: str) -> Optional[int]:
        """
        Gets the value of a specified integer type property.

        Parameters
        ----------
        prop_name : str
            Name of the position attribute.

        Returns
        -------
        Optional[int]
            Integer value if present, otherwise None.
        """
        if self._position is None or not hasattr(self._position, prop_name):
            return None
        value = getattr(self._position, prop_name)
        return None if value is None else int(value)

    def info_double(self, prop_name: str) -> Optional[float]:
        """
        Gets the value of a specified double type property.

        Parameters
        ----------
        prop_name : str
            Name of the position attribute.

        Returns
        -------
        Optional[float]
            Float value if present, otherwise None.
        """
        if self._position is None or not hasattr(self._position, prop_name):
            return None
        value = getattr(self._position, prop_name)
        return None if value is None else float(value)

    def info_string(self, prop_name: str) -> Optional[str]:
        """
        Gets the value of a specified string type property.

        Parameters
        ----------
        prop_name : str
            Name of the position attribute.

        Returns
        -------
        Optional[str]
            String value if present, otherwise None.
        """
        if self._position is None or not hasattr(self._position, prop_name):
            return None
        value = getattr(self._position, prop_name)
        return None if value is None else str(value)

    # ---------- Debug / utility helpers --------------

    def to_dict(self) -> dict[str, Any]:
        """Return all @property values as a dictionary."""
        data: dict[str, Any] = {}
        seen: set[str] = set()

        for cls in type(self).mro():
            for name, attr in vars(cls).items():
                if name in seen:
                    continue
                if isinstance(attr, property):
                    seen.add(name)
                    try:
                        data[name] = getattr(self, name)
                    except Exception as e:
                        data[name] = f"<error: {e}>"

        return data

    def print_all(self) -> None:
        """Print all @property values of the class."""
        for name, value in self.to_dict().items():
            print(f"{name:24} : {value}")

    def __repr__(self) -> str:
        props = self.to_dict()
        props_str = ", ".join(f"{k}={v!r}" for k, v in props.items())
        return f"{self.__class__.__name__}({props_str})"