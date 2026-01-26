    
from typing import Dict
from strategytester5 import *


class TesterConfigValidators:
    """
    Responsible for validating and normalizing strategy tester configurations.
    """

    def __init__(self):
        pass
    
    @staticmethod
    def _validate_keys(raw_config: Dict) -> None:
        
        required_keys = REQUIRED_TESTER_CONFIG_KEYS
        provided_keys = set(raw_config.keys())

        missing = required_keys - provided_keys
        if missing:
            raise RuntimeError(f"Missing tester config keys: {missing}")

        extra = provided_keys - required_keys
        if extra:
            raise RuntimeError(f"Unknown tester config keys: {extra}")
        
    @staticmethod
    def _parse_leverage(leverage: str) -> int:
        """
        Converts '1:100' -> 100
        """
        try:
            left, right = leverage.split(":")
            if left != "1":
                raise ValueError
            value = int(right)
            if value <= 0:
                raise ValueError
            return value
        except Exception:
            raise RuntimeError(f"Invalid leverage format: {leverage}")

    @staticmethod
    def parse_tester_configs(raw_config: Dict) -> Dict:
        TesterConfigValidators._validate_keys(raw_config)

        cfg: Dict = {}

        # --- BOT NAME ---
        cfg["bot_name"] = str(raw_config["bot_name"])

        # --- SYMBOLS ---
        symbols = raw_config["symbols"]
        if not isinstance(symbols, list) or not symbols:
            raise RuntimeError("symbols must be a non-empty list")
        cfg["symbols"] = symbols

        # --- TIMEFRAME ---
        timeframe = raw_config["timeframe"].upper()
        if timeframe not in STRING2TIMEFRAME_MAP:
            raise RuntimeError(f"Invalid timeframe: {timeframe}")
        cfg["timeframe"] = timeframe

        # --- MODELLING ---
        modelling = raw_config["modelling"].lower()
        
        if modelling not in SUPPORTED_TESTER_MODELLING:
            raise RuntimeError(f"Invalid modelling mode: {modelling}, supported modellings include: {SUPPORTED_TESTER_MODELLING}")
        
        cfg["modelling"] = modelling

        # --- DATE PARSING ---
        try:
            start_date = datetime.strptime(
                raw_config["start_date"], "%d.%m.%Y %H:%M"
            )
            end_date = datetime.strptime(
                raw_config["end_date"], "%d.%m.%Y %H:%M"
            )
        except ValueError:
            raise RuntimeError("Date format must be: DD.MM.YYYY HH:MM")

        if start_date >= end_date:
            raise RuntimeError("start_date must be earlier than end_date")

        cfg["start_date"] = start_date
        cfg["end_date"] = end_date

        # --- DEPOSIT ---
        deposit = float(raw_config["deposit"])
        if deposit <= 0:
            raise RuntimeError("deposit must be > 0")
        cfg["deposit"] = deposit

        # --- LEVERAGE ---
        cfg["leverage"] = TesterConfigValidators._parse_leverage(raw_config["leverage"])

        return cfg
