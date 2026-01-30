from strategytester5 import *

class CTrade:
    
    def __init__(self, simulator, magic_number: int, filling_type_symbol: str, deviation_points: int):
        
        self.simulator = simulator
        self.mt5_instance = simulator.mt5_instance
        self.magic_number = magic_number
        self.deviation_points = deviation_points
        self.filling_type = self._get_type_filling(filling_type_symbol)
        
        if self.filling_type == -1:
            print("Failed to initialize the class, Invalid filling type. Check your symbol")
            return
        
    def _get_type_filling(self, symbol):
        
        symbol_info = self.simulator.symbol_info(symbol)
        if not symbol_info:
            print(f"Failed to get symbol info for {symbol}")
            return self.mt5_instance.ORDER_FILLING_FOK
        
        filling_map = {
            1: self.mt5_instance.ORDER_FILLING_FOK,
            2: self.mt5_instance.ORDER_FILLING_IOC,
            4: self.mt5_instance.ORDER_FILLING_BOC,
            8: self.mt5_instance.ORDER_FILLING_RETURN
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
            "action": self.mt5_instance.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": self.deviation_points,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": self.mt5_instance.ORDER_TIME_GTC,
            "type_filling":  self.filling_type,
        }
        
        if sl > 0.0:
            request["sl"] = sl
        if tp > 0.0:
            request["tp"] = tp
        
        if self.simulator.order_send(request) is None:
            return False
        
        # self.simulator.logger.info(f"Position Opened successfully!")
            
        return True
    
    
    def order_open(self, symbol: str, volume: float, order_type: int, price: float, sl: float = 0.0, tp: float = 0.0, type_time: int = MetaTrader5.ORDER_TIME_GTC, expiration: datetime = None, comment: str = "") -> bool:
        
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
        if type_time in (self.mt5_instance.ORDER_TIME_SPECIFIED, self.mt5_instance.ORDER_TIME_SPECIFIED_DAY) and expiration is None:
            print(f"Expiration required for order type {type_time}")
            return False
        
        request = {
            "action": self.mt5_instance.TRADE_ACTION_PENDING,
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
        if type_time in (self.mt5_instance.ORDER_TIME_SPECIFIED, self.mt5_instance.ORDER_TIME_SPECIFIED_DAY) and expiration is not None:
            
            # Convert to broker's expected format (UTC timestamp in milliseconds)
            
            expiration_utc = expiration.astimezone(timezone.utc) if expiration.tzinfo else expiration.replace(tzinfo=timezone.utc)
            request["expiration"] = int(expiration_utc.timestamp() * 1000)
            
            
        # Send order
        
        if self.simulator.order_send(request) is None:
            return False
        
        self.simulator.logger.info(f"Order opened successfully!")
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
    
        return self.position_open(symbol=symbol, volume=volume, order_type=self.mt5_instance.ORDER_TYPE_BUY, price=price, sl=sl, tp=tp, comment=comment)

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
        
        return self.position_open(symbol=symbol, volume=volume, order_type=self.mt5_instance.ORDER_TYPE_SELL, price=price, sl=sl, tp=tp, comment=comment)
    
    def buy_limit(self, volume: float, price: float, symbol: str, sl: float=0.0, tp: float=0.0, type_time: float=MetaTrader5.ORDER_TIME_GTC, expiration: datetime=None, comment: str="") -> bool:
        
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
        
        return self.order_open(symbol=symbol, volume=volume, order_type=self.mt5_instance.ORDER_TYPE_BUY_LIMIT, price=price, sl=sl, tp=tp, type_time=type_time, expiration=expiration, comment=comment)
        
    def sell_limit(self, volume: float, price: float, symbol: str, sl: float=0.0, tp: float=0.0, type_time: float=MetaTrader5.ORDER_TIME_GTC, expiration: datetime=None, comment: str="") -> bool:
            
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

        return self.order_open(symbol=symbol, volume=volume, order_type=self.mt5_instance.ORDER_TYPE_SELL_LIMIT, price=price, sl=sl, tp=tp, type_time=type_time, expiration=expiration, comment=comment)
        
    def buy_stop(self, volume: float, price: float, symbol: str, sl: float=0.0, tp: float=0.0, type_time: float=MetaTrader5.ORDER_TIME_GTC, expiration: datetime=None, comment: str="") -> bool:

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
        
        return self.order_open(symbol=symbol, volume=volume, order_type=self.mt5_instance.ORDER_TYPE_BUY_STOP, price=price, sl=sl, tp=tp, type_time=type_time, expiration=expiration, comment=comment)
        
    def sell_stop(self, volume: float, price: float, symbol: str, sl: float=0.0, tp: float=0.0, type_time: float=MetaTrader5.ORDER_TIME_GTC, expiration: datetime=None, comment: str="") -> bool:
        
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
        
        return self.order_open(symbol=symbol, volume=volume, order_type=self.mt5_instance.ORDER_TYPE_SELL_STOP, price=price, sl=sl, tp=tp, type_time=type_time, expiration=expiration, comment=comment)
    
    """    
    def buy_stop_limit(self, volume: float, price: float, symbol: str, sl: float=0.0, tp: float=0.0, type_time: float=MetaTrader5.ORDER_TIME_GTC, expiration: datetime=None, comment: str="") -> bool:
        
        \"""
        Places a buy stop limit pending order.
        
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
        \"""
        
        return self.order_open(symbol=symbol, volume=volume, order_type=self.mt5_instance.ORDER_TYPE_BUY_STOP_LIMIT, price=price, sl=sl, tp=tp, type_time=type_time, expiration=expiration, comment=comment)
        
    def sell_stop_limit(self, volume: float, price: float, symbol: str, sl: float=0.0, tp: float=0.0, type_time: float=MetaTrader5.ORDER_TIME_GTC, expiration: datetime=None, comment: str="") -> bool:
        
        \"""
        Places a sell stop limit pending order.
        
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
        \"""
        
        return self.order_open(symbol=symbol, volume=volume, order_type=self.mt5_instance.ORDER_TYPE_SELL_STOP_LIMIT, price=price, sl=sl, tp=tp, type_time=type_time, expiration=expiration, comment=comment)

        """
        
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
        if not self.simulator.positions_get(ticket=ticket):
            print(f"Position with ticket {ticket} not found.")
            return False

        position = self.simulator.positions_get(ticket=ticket)[0]
        symbol = position.symbol
        volume = position.volume
        position_type = position.type  # 0=BUY, 1=SELL
    
        # Get close price (BID for buy, ASK for sell)
        
        # print("pos_type: ", position_type, " order type buy: ",self.mt5_instance.ORDER_TYPE_BUY, " order type sell: ",self.mt5_instance.ORDER_TYPE_SELL)
        
        tick_info = self.simulator.symbol_info_tick(symbol)
        price = tick_info.bid if position_type == self.mt5_instance.POSITION_TYPE_BUY else tick_info.ask

        # Set close order type
        order_type = self.mt5_instance.ORDER_TYPE_SELL if position_type == self.mt5_instance.POSITION_TYPE_BUY else self.mt5_instance.ORDER_TYPE_BUY

        request = {
            "action": self.mt5_instance.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": volume,
            "magic": self.magic_number,
            "type": order_type,
            "price": price,
            "deviation": deviation if not isinstance(deviation, float) or not str(deviation) == 'nan' else self.deviation_points, 
            "type_time": self.mt5_instance.ORDER_TIME_GTC,
            "type_filling": self.filling_type,
        }

        # Send the close request
        
        if self.simulator.order_send(request) is None:
            return False

        self.simulator.logger.info(f"Position {ticket} closed successfully!")
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
    
        order = self.simulator.orders_get(ticket=ticket)[0]
        if order is None:
            self.simulator.logger.info(f"Order {order} not found!")
        
        request = {
            "action": self.mt5_instance.TRADE_ACTION_REMOVE,
            "order": ticket,
            "magic": self.magic_number,
            "symbol": order.symbol
        }
        
        # Send the delete request
        
        if self.simulator.order_send(request) is None:
            return False

        self.simulator.logger.info(f"Order {ticket} deleted successfully!")
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
        if not self.simulator.positions_get(ticket=ticket):
            print(f"Position with ticket {ticket} not found.")
            return False

        position = self.simulator.positions_get(ticket=ticket)[0]
        symbol = position.symbol
        
        request = {
            "action": self.mt5_instance.TRADE_ACTION_SLTP,
            "position": ticket,
            "magic": self.magic_number,
            "symbol": symbol,
            "sl": sl,
            "tp": tp
        }
        
        # send a position modify request
        
        if self.simulator.order_send(request) is None:
            return False
        
        self.simulator.logger.info(f"Position {ticket} modified successfully!")
        return True
    
    def order_modify(self, ticket: int, price: float, sl: float, tp: float, type_time: int = MetaTrader5.ORDER_TIME_GTC, expiration: datetime = None, stoplimit: float = 0.0) -> bool:
        
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
        order = self.simulator.orders_get(ticket=ticket)
        if not order:
            print(f"Order with ticket {ticket} not found")
            return False
        
        order = order[0]  # Get the first (and only) order
        
        request = {
            "action": self.mt5_instance.TRADE_ACTION_MODIFY,
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
        if type_time == self.mt5_instance.ORDER_TIME_SPECIFIED:
            if expiration is None:
                print("Error: expiration must be specified for ORDER_TIME_SPECIFIED")
                return False
            
            request["expiration"] = expiration
        
        # Add stoplimit for STOP_LIMIT orders
        if order.type in (self.mt5_instance.ORDER_TYPE_BUY_STOP_LIMIT, self.mt5_instance.ORDER_TYPE_SELL_STOP_LIMIT):
            request["stoplimit"] = stoplimit

        # Send the modification request
        
        if self.simulator.order_send(request) is None:
            return False

        self.simulator.logger.info(f"Order {ticket} modified successfully!")
        return True