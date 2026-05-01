
from __future__ import annotations
import logging

from datetime import datetime, timezone
from typing import Union
from strategytester5.MetaTrader5.api import OverLoadedMetaTrader5API
import MetaTrader5

class CTrade:
    """
        This class resembles the MQL5 Standard Library class `CTrade` and provides
        a convenient Python interface for sending and managing trade requests through
        the MetaTrader 5 Python API.

        Reference (MQL5): https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade
    """

    def __init__(
            self,
            magic_number: int,
            filling_type_symbol: str,
            deviation_points: int,
            terminal: Union[OverLoadedMetaTrader5API|MetaTrader5],
            logger: logging.Logger | None = None
    ):
        """
        Initializes a CTrade wrapper for MetaTrader 5 trade operations.

        Parameters
        ----------
        terminal : Any
            MetaTrader5 module-like or the overloaded/simulated MetaTrader5 instance

        magic_number : int
            Expert Advisor identifier used to tag and track orders and positions
            created by this trade object.

        filling_type_symbol : str
            Symbol name used to determine the appropriate order filling policy
            through the internal `_get_type_filling()` helper.

        deviation_points : int
            Maximum allowed price deviation, in points, when executing market
            orders.

        logger : logging.Logger | None, optional
            Logger instance used for diagnostic and error messages. If None,
            logging output is handled only by the class' internal logic.

        Notes
        -----
        - The constructor resolves and stores the filling type for the provided
          symbol at initialization time.
        - If the filling type cannot be resolved, initialization logs a critical
          error and returns early.
        - This class is intended to act similarly to MQL5's `CTrade`, where trade
          settings such as magic number, filling mode, and deviation are configured
          once and reused across requests.
        """
        
        self.logger = logger

        self.terminal = terminal

        self.magic_number = magic_number
        self.deviation_points = deviation_points
        self.filling_type = self._get_type_filling(filling_type_symbol)
        
        if self.filling_type == -1:
            self._critical_log("Failed to initialize the class, Invalid filling type. Check your symbol")
            return
    
    def _critical_log(self, message: str):
        if self.logger:
            self.logger.critical(message)
        else:
            print(f"CRITICAL: {message}")
            
    def _info_log(self, message: str):
        if self.logger:
            self.logger.info(message)
        else:
            print(f"INFO: {message}")
    
    def _error_log(self, message: str):
        if self.logger:
            self.logger.error(message)
        else:
            print(f"ERROR: {message}")

    def _error_code_log(self, ret_code: int):
        ret_code_description = self.terminal.RETCODE_MAP.get(ret_code, f"Unknown")
        error_message = f"Trade operation failed with retcode {ret_code}: {ret_code_description}"
        return self._error_log(error_message)
    
    def _warning_log(self, message: str):
        if self.logger:
            self.logger.warning(message)
        else:
            print(f"WARNING: {message}")
    
    def _get_type_filling(self, symbol):
        
        symbol_info = self.terminal.symbol_info(symbol)
        if symbol_info is None:
            self._warning_log(f"Failed to get symbol info for {symbol}")
        
        filling_map = {
            1: self.terminal.ORDER_FILLING_FOK,
            2: self.terminal.ORDER_FILLING_IOC,
            4: self.terminal.ORDER_FILLING_BOC,
            8: self.terminal.ORDER_FILLING_RETURN
        }
        
        return filling_map.get(symbol_info.filling_mode, f"Unknown Filling type")
    
    def position_open(self, symbol: str, volume: float, order_type: int, price: float, sl: float=0.0, tp: float=0.0, comment: str="") -> bool:
        
        """
        Open a market position (instant execution).
        
        Executes either a buy or sell order at the current market price. This is for immediate
        position opening, not pending orders.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD", "GBPUSD")
            volume: Trade volume in lots (e.g., 0.1 for micro lot)
            order_type: Trade direction (either ORDER_TYPE_BUY or ORDER_TYPE_SELL)
            price: Execution price. For market orders, this should be the current:
                - Ask price for BUY orders
                - Bid price for SELL orders
            sl: Stop loss price (set to 0.0 to disable)
            tp: Take profit price (set to 0.0 to disable)
            comment: Optional order comment (max 31 characters, will be truncated automatically)
        
        Returns:
            bool: True if position was opened successfully, False otherwise
        """
        
        request = {
            "action": self.terminal.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": self.deviation_points,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": self.terminal.ORDER_TIME_GTC,
            "type_filling":  self.filling_type,
        }
        
        if sl > 0.0:
            request["sl"] = sl
        if tp > 0.0:
            request["tp"] = tp

        result = self.terminal.order_send(request)
        if result.retcode != self.terminal.TRADE_RETCODE_DONE:
            self._error_code_log(result.retcode)
            return False
        
        self._info_log(f"Position #{result.deal} Opened successfully!")
        return True
    
    
    def order_open(self, symbol: str, volume: float, order_type: int, price: float, sl: float = 0.0, tp: float = 0.0, type_time: int = 0, expiration: datetime = None, comment: str = "") -> bool:
        
        """
        Opens a pending order with full control over order parameters.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            volume: Order volume in lots
            order_type: Order type (ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_SELL_STOP, etc.)
            price: Activation price for pending order
            sl: Stop loss price (0 to disable)
            tp: Take profit price (0 to disable)
            type_time: Order expiration type (default: ORDER_TIME_GTC). Possible values:
                    - ORDER_TIME_GTC (Good-Til-Canceled)
                    - ORDER_TIME_DAY (Good for current day)
                    - ORDER_TIME_SPECIFIED (expires at specific datetime)
                    - ORDER_TIME_SPECIFIED_DAY (expires at end of specified day)
            expiration: Expiration datetime (required for ORDER_TIME_SPECIFIED types)
            comment: Optional order comment (max 31 characters)
        
        Returns:
            bool: True if order was placed successfully, False otherwise
        """
        
        # Validate expiration for time-specific orders
        if type_time in (self.terminal.ORDER_TIME_SPECIFIED, self.terminal.ORDER_TIME_SPECIFIED_DAY) and expiration is None:
            self._error_log(f"Expiration required for order type {type_time}")
            return False
        
        request = {
            "action": self.terminal.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.deviation_points,
            "magic": self.magic_number,
            "comment": comment[:31],  # MT5 comment max length is 31 chars
            "type_time": type_time,
            "type_filling": self.filling_type,
        }
        
        # Add expiration if required
        if type_time in (self.terminal.ORDER_TIME_SPECIFIED, self.terminal.ORDER_TIME_SPECIFIED_DAY) and expiration is not None:
            
            # Convert to broker's expected format (UTC timestamp in milliseconds)
            
            expiration_utc = expiration.astimezone(timezone.utc) if expiration.tzinfo else expiration.replace(tzinfo=timezone.utc)
            request["expiration"] = int(expiration_utc.timestamp() * 1000)
            
            
        # Send order
        result = self.terminal.order_send(request)
        if result.retcode != self.terminal.TRADE_RETCODE_DONE:
            self._error_code_log(result.retcode)
            return False

        # self._info_log(f"Order opened successfully!")
        return True
    
    
    def buy(self, volume: float, symbol: str, price: float, sl: float=0.0, tp: float=0.0, comment: str="") -> bool:
        
        """
        Opens a buy (market) position.
        
        Args:
            volume: Trade volume (lot size)
            symbol: Trading symbol (e.g., "EURUSD")
            price: Execution price
            sl: Stop loss price (optional, default=0.0)
            tp: Take profit price (optional, default=0.0)
            comment: Position comment (optional, default="")
        
        Returns:
            bool: True if order was sent successfully, False otherwise
        """
    
        return self.position_open(symbol=symbol, volume=volume, order_type=self.terminal.ORDER_TYPE_BUY, price=price, sl=sl, tp=tp, comment=comment)

    def sell(self, volume: float, symbol: str, price: float, sl: float=0.0, tp: float=0.0, comment: str="") -> bool:
        
        """
        Opens a sell (market) position.
        
        Args:
            volume: Trade volume (lot size)
            symbol: Trading symbol (e.g., "EURUSD")
            price: Execution price
            sl: Stop loss price (optional, default=0.0)
            tp: Take profit price (optional, default=0.0)
            comment: Position comment (optional, default="")
        
        Returns:
            bool: True if order was sent successfully, False otherwise
        """
        
        return self.position_open(symbol=symbol, volume=volume, order_type=self.terminal.ORDER_TYPE_SELL, price=price, sl=sl, tp=tp, comment=comment)
    
    def buy_limit(self, volume: float, price: float, symbol: str, sl: float=0.0, tp: float=0.0, type_time: int=0, expiration: datetime=None, comment: str="") -> bool:
        
        """
        Places a buy limit pending order.
        
        Args:
            volume: Trade volume (lot size)
            price: Execution price
            symbol: Trading symbol (e.g., "EURUSD")
            sl: Stop loss price (optional, default=0.0)
            tp: Take profit price (optional, default=0.0)
            type_time: Order expiration type (default: ORDER_TIME_GTC). Possible values:
                - ORDER_TIME_GTC (Good-Til-Canceled)
                - ORDER_TIME_DAY (Good for current day)
                - ORDER_TIME_SPECIFIED (expires at specific datetime)
                - ORDER_TIME_SPECIFIED_DAY (expires at end of specified day)
            expiration: Expiration datetime (required for ORDER_TIME_SPECIFIED types)
            comment: Order comment (optional, default="")
        
        Returns:
            bool: True if order was placed successfully, False otherwise
        """
        
        return self.order_open(symbol=symbol, volume=volume, order_type=self.terminal.ORDER_TYPE_BUY_LIMIT, price=price, sl=sl, tp=tp, type_time=type_time, expiration=expiration, comment=comment)
        
    def sell_limit(self, volume: float, price: float, symbol: str, sl: float=0.0, tp: float=0.0, type_time: int=0, expiration: datetime=None, comment: str="") -> bool:
            
        """
        Places a sell limit pending order.
        
        Args:
            volume: Trade volume (lot size)
            price: Execution price
            symbol: Trading symbol (e.g., "EURUSD")
            sl: Stop loss price (optional, default=0.0)
            tp: Take profit price (optional, default=0.0)
            type_time: Order expiration type (default: ORDER_TIME_GTC). Possible values:
                - ORDER_TIME_GTC (Good-Til-Canceled)
                - ORDER_TIME_DAY (Good for current day)
                - ORDER_TIME_SPECIFIED (expires at specific datetime)
                - ORDER_TIME_SPECIFIED_DAY (expires at end of specified day)
            expiration: Expiration datetime (required for ORDER_TIME_SPECIFIED types)
            comment: Order comment (optional, default="")
        
        Returns:
            bool: True if order was placed successfully, False otherwise
        """

        return self.order_open(symbol=symbol, volume=volume, order_type=self.terminal.ORDER_TYPE_SELL_LIMIT, price=price, sl=sl, tp=tp, type_time=type_time, expiration=expiration, comment=comment)
        
    def buy_stop(self, volume: float, price: float, symbol: str, sl: float=0.0, tp: float=0.0, type_time: int=0, expiration: datetime=None, comment: str="") -> bool:

        """
        Places a buy stop pending order.
        
        Args:
            volume: Trade volume (lot size)
            price: Execution price
            symbol: Trading symbol (e.g., "EURUSD")
            sl: Stop loss price (optional, default=0.0)
            tp: Take profit price (optional, default=0.0)
            type_time: Order expiration type (default: ORDER_TIME_GTC). Possible values:
                - ORDER_TIME_GTC (Good-Til-Canceled)
                - ORDER_TIME_DAY (Good for current day)
                - ORDER_TIME_SPECIFIED (expires at specific datetime)
                - ORDER_TIME_SPECIFIED_DAY (expires at end of specified day)
            expiration: Expiration datetime (required for ORDER_TIME_SPECIFIED types)
            comment: Order comment (optional, default="")
        
        Returns:
            bool: True if order was placed successfully, False otherwise
        """
        
        return self.order_open(symbol=symbol, volume=volume, order_type=self.terminal.ORDER_TYPE_BUY_STOP, price=price, sl=sl, tp=tp, type_time=type_time, expiration=expiration, comment=comment)
        
    def sell_stop(self, volume: float, price: float, symbol: str, sl: float=0.0, tp: float=0.0, type_time: int=0, expiration: datetime=None, comment: str="") -> bool:
        
        """
        Places a sell stop pending order.
        
        Args:
            volume: Trade volume (lot size)
            price: Execution price
            symbol: Trading symbol (e.g., "EURUSD")
            sl: Stop loss price (optional, default=0.0)
            tp: Take profit price (optional, default=0.0)
            type_time: Order expiration type (default: ORDER_TIME_GTC). Possible values:
                  - ORDER_TIME_GTC (Good-Til-Canceled)
                  - ORDER_TIME_DAY (Good for current day)
                  - ORDER_TIME_SPECIFIED (expires at specific datetime)
                  - ORDER_TIME_SPECIFIED_DAY (expires at end of specified day)
            expiration: Expiration datetime (required for ORDER_TIME_SPECIFIED types)
            comment: Order comment (optional, default="")
        
        Returns:
            bool: True if order was placed successfully, False otherwise
        """
        
        return self.order_open(symbol=symbol, volume=volume, order_type=self.terminal.ORDER_TYPE_SELL_STOP, price=price, sl=sl, tp=tp, type_time=type_time, expiration=expiration, comment=comment)
        
    def position_close(self, ticket: int, deviation: float=float("nan")) -> bool:
        
        """
        Closes an open position by ticket number.
        
        Args:
            ticket: Position ticket number
            deviation: Maximum price deviation in points (optional)
        
        Returns:
            bool: True if position was closed successfully, False otherwise
        
        Raises:
            Prints error message if position not found or close fails
        """
            
        # Select position by ticket
        if not self.terminal.positions_get(ticket=ticket):
            self._warning_log(f"Position with ticket {ticket} not found.")
            return False

        position = self.terminal.positions_get(ticket=ticket)[0]
        symbol = position.symbol
        volume = position.volume
        position_type = position.type  # 0=BUY, 1=SELL
    
        # Get close price (BID for buy, ASK for sell)
        
        tick_info = self.terminal.symbol_info_tick(symbol)
        price = tick_info.bid if position_type == self.terminal.POSITION_TYPE_BUY else tick_info.ask

        # Set close order type
        order_type = self.terminal.ORDER_TYPE_SELL if position_type == self.terminal.POSITION_TYPE_BUY else self.terminal.ORDER_TYPE_BUY

        request = {
            "action": self.terminal.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": volume,
            "magic": self.magic_number,
            "type": order_type,
            "price": price,
            "deviation": deviation if not isinstance(deviation, float) or not str(deviation) == 'nan' else self.deviation_points, 
            "type_time": self.terminal.ORDER_TIME_GTC,
            "type_filling": self.filling_type,
        }

        # Send the close request
        
        if self.terminal.order_send(request) is None:
            return False

        self._info_log(f"Position {ticket} closed successfully!")
        return True
    
    def order_delete(self, ticket: int) -> bool:
        
        """
        Deletes a pending order by ticket number.
        
        Args:
            ticket: Order ticket number
        
        Returns:
            bool: True if order was deleted successfully, False otherwise
        
        Raises:
            Prints error message if deletion fails
        """
    
        order = self.terminal.orders_get(ticket=ticket)[0]
        if order is None:
            self._info_log(f"Order {order} not found!")
        
        request = {
            "action": self.terminal.TRADE_ACTION_REMOVE,
            "order": ticket,
            "magic": self.magic_number,
            "symbol": order.symbol
        }
        
        # Send the delete request
        
        if self.terminal.order_send(request) is None:
            return False

        self._info_log(f"Order {ticket} deleted successfully!")
        return True
            

    def position_modify(self, ticket: int, sl: float, tp: float) -> bool:
        
        """
        Modifies stop loss and take profit of an open position.
        
        Args:
            ticket: Position ticket number
            sl: New stop loss price
            tp: New take profit price
        
        Returns:
            bool: True if modification was successful, False otherwise
        
        Raises:
            Prints error message if position not found or modification fails
        """
        
        # Select position by ticket
        if not self.terminal.positions_get(ticket=ticket):
            self._warning_log(f"Position with ticket {ticket} not found.")
            return False

        position = self.terminal.positions_get(ticket=ticket)[0]
        symbol = position.symbol
        
        request = {
            "action": self.terminal.TRADE_ACTION_SLTP,
            "position": ticket,
            "magic": self.magic_number,
            "symbol": symbol,
            "sl": sl,
            "tp": tp
        }
        
        # send a position modify request
        
        if self.terminal.order_send(request) is None:
            return False
        
        self._info_log(f"Position {ticket} modified successfully!")
        return True
    
    def order_modify(self, ticket: int, price: float, sl: float, tp: float, type_time: int = 0, expiration: datetime = None, stoplimit: float = 0.0) -> bool:
        
        """
        Modify parameters of a pending order with full control similar to MQL5's OrderModify.
        
        Args:
            ticket: Order ticket number
            price: New activation price for the pending order
            sl: New stop loss price (0 to remove)
            tp: New take profit price (0 to remove)
            type_time: Order expiration type (ORDER_TIME_GTC, ORDER_TIME_DAY, etc.)
            expiration: Order expiration time (required for ORDER_TIME_SPECIFIED)
            stoplimit: StopLimit price for STOP_LIMIT orders
        
        Returns:
            bool: True if order was modified successfully, False otherwise
        
        Raises:
            Prints error message if modification fails
        """
        
        # Get the order by ticket
        order = self.terminal.orders_get(ticket=ticket)
        if not order:
            self._warning_log(f"Order with ticket {ticket} not found")
            return False
        
        order = order[0]  # Get the first (and only) order
        
        request = {
            "action": self.terminal.TRADE_ACTION_MODIFY,
            "order": ticket,
            "price": price,
            "sl": sl,
            "tp": tp,
            "symbol": order.symbol,
            "type": order.type,
            "magic": self.magic_number,
            "type_time": type_time,
            "type_filling": self.filling_type,
        }
        
        # Add expiration if specified (for ORDER_TIME_SPECIFIED)
        if type_time == self.terminal.ORDER_TIME_SPECIFIED:
            if expiration is None:
                self._warning_log("Error: expiration must be specified for ORDER_TIME_SPECIFIED")
                return False
            
            request["expiration"] = expiration
        
        # Add stoplimit for STOP_LIMIT orders
        if order.type in (self.terminal.ORDER_TYPE_BUY_STOP_LIMIT, self.terminal.ORDER_TYPE_SELL_STOP_LIMIT):
            request["stoplimit"] = stoplimit

        # Send the modification request

        result = self.terminal.order_send(request=request)
        if result.retcode != self.terminal.TRADE_RETCODE_DONE:
            self._error_code_log(result.retcode)
            return False

        # self._info_log(f"Order {ticket} modified successfully!")
        return True