from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Union
from ..MetaTrader5.constants import MetaTrader5Constants as mt5_constants
from ..MetaTrader5 import TradeOrder

class COrderInfo:
    """
        A lightweight Python wrapper that resembles the MQL5 Standard Library class
        `COrderInfo` and provides convenient, read-only access to the currently
        selected pending/open order properties in MetaTrader 5.
    
        This class acts like a cursor over one selected order stored in `self._order`.
        You may select an order directly with `select_order(order)`, by ticket using
        `select(ticket)`, or by index using `select_by_index(index)`.
    
        [Reference (MQL5)](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo)
    """

    def __init__(self, order: TradeOrder) -> None:
        """
        Instantiates the COrderInfo object (class).

        Args:
            order : A tuple representing an order, typically obtained from `mt5.history_orders_get()` or `mt5.orders_get()`. This should be a structure compatible with the MetaTrader5 Python API's order representation.

        Notes:
            - If no order is selected, properties return `None`, `0`, `0.0`, empty string, or `"N/A"` depending on the property.
            - `time_setup`, `time_done`, and `time_expiration` are returned as timezone-aware UTC datetimes where possible.
            - Description methods intentionally preserve your original mapping logic/style.

        """

        if self.order is None:
            raise ValueError("Order cannot be None. Please provide a valid order tuple.")

        self._order = order

    # --------- Integer / datetime properties -----------

    @property
    def ticket(self) -> int:
        """Gets the ticket of the selected order."""
        return int(self._order.ticket) if self._order else -1

    @property
    def time_setup(self) -> Optional[datetime]:
        """Gets the time of order placement."""
        return datetime.fromtimestamp(self._order.time_setup, tz=timezone.utc) if self._order else None

    @property
    def time_setup_msc(self) -> int:
        """Receives the time of placing an order in milliseconds since 01.01.1970."""
        return int(self._order.time_setup_msc) if self._order else 0

    @property
    def time_done(self) -> Optional[datetime]:
        """Gets the time of order execution or cancellation."""
        if not self._order or int(self._order.time_done) == 0:
            return None
        return datetime.fromtimestamp(self._order.time_done, tz=timezone.utc)

    @property
    def time_done_msc(self) -> int:
        """Receives order execution or cancellation time in milliseconds since 01.01.1970."""
        return int(self._order.time_done_msc) if self._order else 0

    @property
    def order_type(self) -> int:
        """Gets the order type."""
        return int(self._order.type) if self._order else -1

    @property
    def order_type_description(self) -> str:
        """Gets the order type as a string."""

        order_type_map = {
            mt5_constants.ORDER_TYPE_BUY: "Market Buy order",
            mt5_constants.ORDER_TYPE_SELL: "Market Sell order",
            mt5_constants.ORDER_TYPE_BUY_LIMIT: "Buy Limit pending order",
            mt5_constants.ORDER_TYPE_SELL_LIMIT: "Sell Limit pending order",
            mt5_constants.ORDER_TYPE_BUY_STOP: "Buy Stop pending order",
            mt5_constants.ORDER_TYPE_SELL_STOP: "Sell Stop pending order",
            mt5_constants.ORDER_TYPE_BUY_STOP_LIMIT: "Upon reaching the order price, a pending Buy Limit order is placed at the StopLimit price",
            mt5_constants.ORDER_TYPE_SELL_STOP_LIMIT: "Upon reaching the order price, a pending Sell Limit order is placed at the StopLimit price",
            mt5_constants.ORDER_TYPE_CLOSE_BY: "Order to close a position by an opposite one"
        }

        return order_type_map.get(self.order_type, "Unknown order type")

    @property
    def state(self) -> int:
        """Gets the order state."""
        return int(self._order.state) if self._order else -1

    @property
    def state_description(self) -> str:
        """Gets the order state as a string."""

        state_map = {
            mt5_constants.ORDER_STATE_STARTED: "Order checked, but not yet accepted by broker",
            mt5_constants.ORDER_STATE_PLACED: "Order accepted",
            mt5_constants.ORDER_STATE_CANCELED: "Order canceled by client",
            mt5_constants.ORDER_STATE_PARTIAL: "Order partially executed",
            mt5_constants.ORDER_STATE_FILLED: "Order fully executed",
            mt5_constants.ORDER_STATE_REJECTED: "Order rejected",
            mt5_constants.ORDER_STATE_EXPIRED: "Order expired",
            mt5_constants.ORDER_STATE_REQUEST_ADD: "Order is being registered (placing to the trading system)",
            mt5_constants.ORDER_STATE_REQUEST_MODIFY: "Order is being modified (changing its parameters)",
            mt5_constants.ORDER_STATE_REQUEST_CANCEL: "Order is being deleted (deleting from the trading system)"
        }

        return state_map.get(self.state, "Unknown order state")

    @property
    def time_expiration(self) -> Optional[datetime]:
        """Gets the time of order expiration."""
        if not self._order:
            return None

        if self._order.time_expiration == 0:
            return None

        try:
            return datetime.fromtimestamp(self._order.time_expiration, tz=timezone.utc)
        except (ValueError, OSError):
            return None

    @property
    def type_filling(self) -> int:
        """Gets the type of order execution by remainder."""
        return int(self._order.type_filling) if self._order else -1

    @property
    def type_filling_description(self) -> Optional[str]:
        """Gets the type of order execution by remainder as a string."""

        filling_map = {
            1: "FOK (Fill or Kill)",
            2: "IOC (Immediate or Cancel)",
            4: "BOC (Book or Cancel)",
            8: "RETURN"
        }

        return filling_map.get(self.type_filling)

    @property
    def type_time(self) -> int:
        """Gets the type of order at the time of the expiration."""
        return int(self._order.type_time) if self._order else -1

    @property
    def type_time_description(self) -> str:
        """Gets the order type by expiration time as a string."""

        if not self._order:
            return "unknown"

        type_time = self._order.type_time

        if type_time == mt5_constants.ORDER_TIME_SPECIFIED:
            return "ORDER_TIME_SPECIFIED"
        elif type_time == mt5_constants.ORDER_TIME_SPECIFIED_DAY:
            return "ORDER_TIME_SPECIFIED_DAY"
        elif type_time == mt5_constants.ORDER_TIME_DAY:
            return "ORDER_TIME_DAY"
        elif type_time == mt5_constants.ORDER_TIME_GTC:
            return "ORDER_TIME_GTC"
        else:
            return "unknown"

    @property
    def magic(self) -> int:
        """Gets the ID of expert that placed the order."""
        return int(self._order.magic) if self._order else -1

    @property
    def position_id(self) -> int:
        """Gets the ID of position."""
        return int(self._order.position_id) if self._order else -1

    #  Double properties -----------

    @property
    def volume_initial(self) -> float:
        """Gets the initial volume of the order."""
        return float(self._order.volume_initial) if self._order else 0.0

    @property
    def volume_current(self) -> float:
        """Gets the unfilled volume of the order."""
        return float(self._order.volume_current) if self._order else 0.0

    @property
    def price_open(self) -> float:
        """Gets the order price."""
        return float(self._order.price_open) if self._order else 0.0

    @property
    def stop_loss(self) -> float:
        """Gets the order Stop Loss."""
        return float(self._order.sl) if self._order else 0.0

    @property
    def take_profit(self) -> float:
        """Gets the order Take Profit."""
        return float(self._order.tp) if self._order else 0.0

    @property
    def price_current(self) -> float:
        """Gets the current price by order symbol."""
        if not self._order:
            return 0.0

        tick = mt5_constants.symbol_info_tick(self._order.symbol)
        if not tick:
            return 0.0

        return tick.bid if self._order.type in [
            mt5_constants.ORDER_TYPE_BUY,
            mt5_constants.ORDER_TYPE_BUY_LIMIT,
            mt5_constants.ORDER_TYPE_BUY_STOP
        ] else tick.ask

    @property
    def price_stop_limit(self) -> float:
        """Gets the price of a Limit order."""
        return float(self._order.price_stoplimit) if self._order else 0.0

    #  String properties ------------

    @property
    def symbol(self) -> str:
        """Gets the name of the order symbol."""
        return str(self._order.symbol) if self._order else ""

    @property
    def comment(self) -> str:
        """Gets the order comment."""
        return str(self._order.comment) if self._order else ""

    @property
    def external_id(self) -> str:
        """Gets the external ID."""
        value = getattr(self._order, "external_id", "")
        return str(value) if self._order else ""

    # --------- Generic Info* methods -----------

    def info_integer(self, prop_name: str) -> Optional[int]:
        """
        Gets the value of a specified integer type property.

        Args:
            prop_name : Name of the order attribute.

        Returns:
            Integer value if present, otherwise None.
        """

        if self._order is None or not hasattr(self._order, prop_name):
            return None
        value = getattr(self._order, prop_name)
        return None if value is None else int(value)

    def info_double(self, prop_name: str) -> Optional[float]:
        """
        Gets the value of a specified double type property.

        Args:
            prop_name : Name of the order attribute.

        Returns:
            Float value if present, otherwise None.
        """
        if self._order is None or not hasattr(self._order, prop_name):
            return None
        value = getattr(self._order, prop_name)
        return None if value is None else float(value)

    def info_string(self, prop_name: str) -> Optional[str]:
        """
        Gets the value of a specified string type property.

        Args:
            prop_name: Name of the order attribute.

        Returns:
            String value if present, otherwise None.
        """
        
        if self._order is None or not hasattr(self._order, prop_name):
            return None
        value = getattr(self._order, prop_name)
        return None if value is None else str(value)

    # --------- Debug / utility helpers ---------

    def to_dict(self) -> dict[str, Any]:
        """Return all @property values from the class and its parents."""
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