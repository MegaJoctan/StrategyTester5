from __future__ import annotations

from typing import Dict
from strategytester5.MetaTrader5.api import VirtualMetaTrader5
from . import config
from datetime import datetime
from dateutil import parser


class TesterConfigValidators:
    """
        A module responsible for validating and normalizing strategy tester configurations.
    """

    def __init__(self):
        pass

    @staticmethod
    def _validate_keys(raw_config: Dict) -> None:

        required_keys = config.REQUIRED_TESTER_CONFIG_KEYS
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
    def _parse_modelling(value):
        # already integer
        if isinstance(value, int):
            if value not in config.SUPPORTED_TESTER_MODELLING:
                raise RuntimeError(f"Invalid modelling integer: {value}")
            return value

        # string input
        if isinstance(value, str):
            key = value.lower()
            if key not in config.SUPPORTED_TESTER_MODELLING_REVERSE:
                raise RuntimeError(
                    f"Invalid modelling: {value}, supported: {list(config.SUPPORTED_TESTER_MODELLING.values())}"
                )
            return config.SUPPORTED_TESTER_MODELLING_REVERSE[key]

        raise RuntimeError(f"Invalid modelling type: {type(value)}")

    @staticmethod
    def parse_date(date_str: str) -> datetime:
        try:
            return parser.parse(date_str, dayfirst=True)
        except Exception:
            raise RuntimeError(f"Invalid date format: '{date_str}'")

    @staticmethod
    def parse_tester_configs(raw_config: Dict) -> Dict:

        """ Validates and normalizes raw tester configuration dictionary. """

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
        timeframe = raw_config["timeframe"]
        if timeframe not in VirtualMetaTrader5.STRING2TIMEFRAME_MAP:
            raise RuntimeError(
                f"Invalid timeframe: {timeframe} supported: {VirtualMetaTrader5.STRING2TIMEFRAME_MAP.keys()}")
        cfg["timeframe"] = timeframe

        # --- MODELLING ---

        cfg["modelling"] = TesterConfigValidators._parse_modelling(raw_config["modelling"])

        # --- DATE PARSING ---

        start_date = TesterConfigValidators.parse_date(raw_config["start_date"])
        end_date = TesterConfigValidators.parse_date(raw_config["end_date"])

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
