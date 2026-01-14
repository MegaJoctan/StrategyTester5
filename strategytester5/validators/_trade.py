from strategytester5 import *

class TradeValidators:
    def __init__(self, 
                symbol_info: SymbolInfo, 
                logger: any,
                mt5_instance: MetaTrader5):
        
        self.symbol_info = symbol_info
        self.logger = logger
        self.mt5_instance = mt5_instance
        
    def is_valid_lotsize(self, lotsize: float) -> bool:
        
        # Validate lotsize
        
        if lotsize < self.symbol_info.volume_min: # check if the received lotsize is smaller than minimum accepted lot of a symbol
            self.logger.info(f"Trade validation failed: lotsize ({lotsize}) is less than minimum allowed ({self.symbol_info.volume_min})")
            return False
        
        if lotsize > self.symbol_info.volume_max: # check if the received lotsize is greater than the maximum accepted lot
            self.logger.info(f"Trade validation failed: lotsize ({lotsize}) is greater than maximum allowed ({self.symbol_info.volume_max})")
            return False
        
        step_count = lotsize / self.symbol_info.volume_step 
        
        if abs(step_count - round(step_count)) > 1e-7: # check if the stoploss is a multiple of the step size
            self.logger.info(f"Trade validation failed: lotsize ({lotsize}) must be a multiple of step size ({self.symbol_info.volume_step})")
            return False

        return True
    
    def is_valid_freeze_level(self, tick_info: Tick, entry: float, stop_price: float, order_type: int) -> bool:
        """
        Check SYMBOL_TRADE_FREEZE_LEVEL for pending orders and open positions.
        """

        freeze_level = self.symbol_info.trade_freeze_level
        if freeze_level <= 0:
            return True  # No freeze restriction

        point = self.symbol_info.point
        freeze_distance = freeze_level * point

        bid = tick_info.bid
        ask = tick_info.ask

        def log_fail(msg: str, dist: float):
            self.logger.info(
                f"{msg} | distance={dist/point:.1f} pts < "
                f"freeze_level={freeze_level} pts"
            )

        # ---------------- Pending Orders ----------------

        if order_type == self.mt5_instance.ORDER_TYPE_BUY_LIMIT:
            dist = ask - entry
            if dist < freeze_distance:
                log_fail("BuyLimit cannot be modified: Ask - OpenPrice", dist)
                return False
            return True

        if order_type == self.mt5_instance.ORDER_TYPE_SELL_LIMIT:
            dist = entry - bid
            if dist < freeze_distance:
                log_fail("SellLimit cannot be modified: OpenPrice - Bid", dist)
                return False
            return True

        if order_type == self.mt5_instance.ORDER_TYPE_BUY_STOP:
            dist = entry - ask
            if dist < freeze_distance:
                log_fail("BuyStop cannot be modified: OpenPrice - Ask", dist)
                return False
            return True

        if order_type == self.mt5_instance.ORDER_TYPE_SELL_STOP:
            dist = bid - entry
            if dist < freeze_distance:
                log_fail("SellStop cannot be modified: Bid - OpenPrice", dist)
                return False
            return True

        # ---------------- Open Positions (SL / TP modification) ----------------

        # Buy position
        if order_type == self.mt5_instance.ORDER_TYPE_BUY:
            if stop_price <= 0:
                return True

            if stop_price < entry:  # StopLoss
                dist = bid - stop_price
                if dist < freeze_distance:
                    log_fail("Buy position SL cannot be modified: Bid - SL", dist)
                    return False
            else:  # TakeProfit
                dist = stop_price - bid
                if dist < freeze_distance:
                    log_fail("Buy position TP cannot be modified: TP - Bid", dist)
                    return False

            return True

        # Sell position
        if order_type == self.mt5_instance.ORDER_TYPE_SELL:
            if stop_price <= 0:
                return True

            if stop_price > entry:  # StopLoss
                dist = stop_price - ask
                if dist < freeze_distance:
                    log_fail("Sell position SL cannot be modified: SL - Ask", dist)
                    return False
            else:  # TakeProfit
                dist = ask - stop_price
                if dist < freeze_distance:
                    log_fail("Sell position TP cannot be modified: Ask - TP", dist)
                    return False

            return True

        self.logger.error("Unknown MetaTrader 5 order type")
        return False
    
    def is_max_orders_reached(self, open_orders: int, ac_limit_orders: int) -> bool:
        """Checks whether the maximum number of orders for the account is reached

        Args:
            open_orders (int): The number of opened orders
            ac_limit_orders (int): Maximum number of orders allowed for the account

        Returns:
            bool: True if the threshold is reached, otherwise, it returns false.
        """
        
        if open_orders >= ac_limit_orders and ac_limit_orders > 0:
            self.logger.critical(f"Pending Orders limit of {ac_limit_orders} is reached!")
            return True
        
        return False
    
    def is_symbol_volume_reached(self, symbol_volume: float, volume_limit: float) -> bool:
        
        """Checks if the maximum allowed volume is reached for a particular instrument

        Returns:
            bool: True if the condition is reached and False when it is not.
        """
    
        if symbol_volume >= volume_limit and volume_limit > 0:
            self.logger.critical(f"Symbol Volume limit of {volume_limit} is reached!")
            return True
        
        return False
    
    def is_valid_stops_level(self, entry: float, stop_price: float, stops_type: str='') -> bool:
        
        point = self.symbol_info.point
        stop_level   = self.symbol_info.trade_stops_level * point
        
        distance = abs(entry-stop_price)
        
        if stop_price <= 0:
            return True
        
        if distance < stop_level:
            self.logger.info(f"{'Either SL or TP' if stops_type=='' else stops_type} is too close to the market. Min allowed distance = {stop_level}")
            return False
        
        return True
    
    def is_valid_sl(self, entry: float, sl: float, order_type: int) -> bool:
        
        if not self.is_valid_stops_level(entry, sl, "Stoploss"): # check for stops and freeze levels
            return False
            
        if sl > 0:
            if order_type in BUY_ACTIONS: # buy action
                
                if sl >= entry:
                    self.logger.info(f"Trade validation failed: Buy-based order's stop loss ({sl}) must be below order opening price ({entry})")
                    return False
                
            elif order_type in SELL_ACTIONS: # sell action
                
                if sl <= entry:
                    self.logger.info(f"Trade validation failed: Sell-based order's stop loss ({sl}) must be above order opening price ({entry})")
                    return False
            
            else:
                self.logger.error("Unknown MetaTrader 5 order type")
                return False
        
        return True

    def is_valid_tp(self, entry: float, tp: float, order_type: int) -> bool:
        
        if not self.is_valid_stops_level(entry, tp, "Takeprofit"): # check for stops and freeze levels
            return False
        
        if tp > 0:
            if order_type in BUY_ACTIONS: # buy position
                if tp <= entry:
                    self.logger.info(f"Trade validation failed: {ORDER_TYPE_MAP[order_type]} take profit ({tp}) must be above order opening price ({entry})")
                    return False
            elif order_type in SELL_ACTIONS: # sell position
                if tp >= entry:
                    self.logger.info(f"Trade validation failed: {ORDER_TYPE_MAP[order_type]} take profit ({tp}) must be below order opening price ({entry})")
                    return False
            else:
                self.logger.error("Unknown MetaTrader 5 order type")
                return False
        
        return True
    
    @staticmethod    
    def price_equal(a: float, b: float, eps: float = 1e-8) -> bool:
        return abs(a - b) <= eps

    def is_valid_entry_price(self, tick_info: Tick, price: float, order_type: int) -> bool:
        
        eps = pow(10, -self.symbol_info.digits)
        if order_type == self.mt5_instance.ORDER_TYPE_BUY:  # BUY
            if not self.price_equal(a=price, b=tick_info.ask, eps=eps):
                self.logger.info(f"Trade validation failed: Buy price {price} != ask {tick_info.ask}")
                return False

        elif order_type == self.mt5_instance.ORDER_TYPE_SELL:  # SELL
            if not self.price_equal(a=price, b=tick_info.bid, eps=eps):
                self.logger.info(f"Trade validation failed: Sell price {price} != bid {tick_info.bid}")
                return False
        else:
            self.logger.error("Unknown MetaTrader 5 position type")
            return False

        return True
    
    def is_there_enough_money(self, margin_required: float, free_margin: float) -> bool:
        
        if margin_required < 0:
            self.logger.info("Trade validation failed: Cannot calculate margin requirements")
            return False
        
        # Check free margin
        if margin_required > free_margin:
            self.logger.info(f'Trade validation failed: Not enough money to open trade. '
                f'Required: {margin_required:.2f}, '
                f'Free margin: {free_margin:.2f}')
            
            return False

        return True
