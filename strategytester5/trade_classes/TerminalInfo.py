from __future__ import annotations

from typing import Any, Optional, Union
from . import StrategyTester
from .. import MetaTrader5

class CTerminalInfo:
    """
        A lightweight Python wrapper that resembles the MQL5 Standard Library class
        `CTerminalInfo` and provides convenient, read-only access to the properties
        of the MetaTrader 5 terminal environment.

        This class caches the result of `mt5.terminal_info()` at construction time.
        The returned values reflect the terminal state at the moment of initialization.
        If you need up-to-date values, create a new instance or add a refresh method.

        [Reference (MQL5)](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo)
    """

    def __init__(self, terminal: Union[StrategyTester|MetaTrader5]):
        
        """
        Instantiates the TerminalInfo object.

        Args:
            terminal : MetaTrader5 module-like or the StrategyTester instance

        Raises:
            RuntimeError: If terminal information cannot be retrieved.

        Notes:

            Method groups mirror the MQL5 layout:
            - Integer / boolean properties: Build, IsConnected, IsDLLsAllowed, IsTradeAllowed, etc.
            - String properties: Language, Name, Company, Path, DataPath, CommonDataPath
            - Generic accessors: InfoInteger, InfoString
        """

        self.terminal = terminal
        if isinstance(terminal, StrategyTester):
            self.terminal = terminal.mt5_instance

        self._info = self.terminal.terminal_info()

        if self._info is None:
            raise RuntimeError("Failed to retrieve terminal info")

    # ------------- State / boolean properties -------------

    @property
    def is_valid(self) -> bool:
        """Returns True if terminal info is available."""
        return self._info is not None

    @property
    def is_connected(self) -> bool:
        """Gets the information about connection to the trade server."""
        return bool(self._info.connected)

    @property
    def is_dlls_allowed(self) -> bool:
        """Gets the information about permission of DLL usage."""
        return bool(self._info.dlls_allowed)

    @property
    def is_trade_allowed(self) -> bool:
        """Gets the information about permission to trade."""
        return bool(self._info.trade_allowed)

    @property
    def is_email_enabled(self) -> bool:
        """Gets the information about permission to send emails."""
        return bool(self._info.email_enabled)

    @property
    def is_ftp_enabled(self) -> bool:
        """Gets the information about permission to send FTP reports."""
        return bool(self._info.ftp_enabled)

    @property
    def is_community_account(self) -> bool:
        """Gets whether an MQL5.community account is configured."""
        return bool(self._info.community_account)

    @property
    def is_community_connection(self) -> bool:
        """Gets whether the terminal is connected to the MQL5.community service."""
        return bool(self._info.community_connection)

    @property
    def are_notifications_enabled(self) -> bool:
        """Gets whether push notifications are enabled."""
        return bool(self._info.notifications_enabled)

    @property
    def is_mqid(self) -> bool:
        """Gets whether a MetaQuotes ID is configured."""
        return bool(self._info.mqid)

    @property
    def is_tradeapi_disabled(self) -> bool:
        """Gets whether external trade API usage is disabled."""
        return bool(self._info.tradeapi_disabled)

    @property
    def is_x64(self) -> bool:
        """Gets the information about the type of the client terminal."""
        value = getattr(self._info, "x64", False)
        return bool(value)

    # ---------- Integer / numeric properties ------------

    @property
    def build(self) -> int:
        """Gets the build number of the client terminal."""
        return int(self._info.build)

    @property
    def max_bars(self) -> int:
        """Gets the maximum number of bars on chart."""
        return int(self._info.maxbars)

    @property
    def code_page(self) -> int:
        """Gets the code page of the terminal language."""
        return int(self._info.codepage)

    @property
    def cpu_cores(self) -> int:
        """Gets the number of CPU cores."""
        value = getattr(self._info, "cpu_cores", 0)
        return int(value)

    @property
    def memory_physical(self) -> int:
        """Gets the amount of physical memory in MB."""
        value = getattr(self._info, "memory_physical", 0)
        return int(value)

    @property
    def memory_total(self) -> int:
        """Gets the total memory available for terminal/agent process in MB."""
        value = getattr(self._info, "memory_total", 0)
        return int(value)

    @property
    def memory_available(self) -> int:
        """Gets the free memory available for terminal/agent process in MB."""
        value = getattr(self._info, "memory_available", 0)
        return int(value)

    @property
    def memory_used(self) -> int:
        """Gets the memory used by terminal/agent process in MB."""
        value = getattr(self._info, "memory_used", 0)
        return int(value)

    @property
    def disk_space(self) -> int:
        """Gets the free disk space in MB."""
        value = getattr(self._info, "disk_space", 0)
        return int(value)

    @property
    def ping_last(self) -> int:
        """Gets the last ping value."""
        return int(self._info.ping_last)

    @property
    def community_balance(self) -> float:
        """Gets the MQL5.community balance."""
        return float(self._info.community_balance)

    @property
    def retransmission(self) -> float:
        """Gets the retransmission percentage/value."""
        return float(self._info.retransmission)

    @property
    def opencl_support(self) -> str:
        """Gets the OpenCL version supported by the video card."""
        value = getattr(self._info, "opencl_support", "")
        return str(value)

    # ---------- String properties --------------

    @property
    def name(self) -> str:
        """Gets the name of the client terminal."""
        return str(self._info.name)

    @property
    def company(self) -> str:
        """Gets the company name of the client terminal."""
        return str(self._info.company)

    @property
    def language(self) -> str:
        """Gets the language of the client terminal."""
        return str(self._info.language)

    @property
    def path(self) -> str:
        """Gets the folder of the client terminal."""
        return str(self._info.path)

    @property
    def data_path(self) -> str:
        """Gets the data folder of the client terminal."""
        return str(self._info.data_path)

    @property
    def common_data_path(self) -> str:
        """Gets the common data folder of all installed terminals."""
        return str(self._info.commondata_path)

    # ------------- Generic Info* methods ------------

    def info_integer(self, prop_name: str) -> Optional[int]:
        """
        Gets the value of a property of integer type.

        Args:
            prop_name : Name of the terminal info attribute.

        Returns:
            Integer value if present, otherwise None.
        """
        if self._info is None or not hasattr(self._info, prop_name):
            return None

        value = getattr(self._info, prop_name)
        return None if value is None else int(value)

    def info_string(self, prop_name: str) -> Optional[str]:
        """
        Gets the value of a property of string type.

        Args:
            prop_name : Name of the terminal info attribute.

        Returns:
            String value if present, otherwise None.
        """
        if self._info is None or not hasattr(self._info, prop_name):
            return None

        value = getattr(self._info, prop_name)
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