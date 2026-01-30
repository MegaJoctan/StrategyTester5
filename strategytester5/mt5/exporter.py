import MetaTrader5
from pathlib import Path
import os
import json

def get_server_name(mt5_instance: MetaTrader5) -> str:

    ac_info = mt5_instance.account_info()
    if ac_info is None:
        raise RuntimeError(f"account_info() failed: {mt5_instance.last_error()}")

    return ac_info.server

def all_symbol_info(mt5_instance: MetaTrader5, out_path: str):
    symbols = mt5_instance.symbols_get()
    if symbols is None:
        raise RuntimeError(f"symbols_get() failed: {mt5_instance.last_error()}")

    exported = []
    skipped = []

    for s in symbols:
        name = getattr(s, "name", None)
        if not name:
            continue

        info = mt5_instance.symbol_info(name)
        if info is None:
            skipped.append({"symbol": name, "reason": "symbol_info returned None"})
            continue

        # SymbolInfo is a namedtuple-like object
        d = info._asdict() if hasattr(info, "_asdict") else dict(info)

        exported.append(d)

    payload = {
        "exported": len(exported),
        "skipped": skipped,
        "symbols": exported,
    }

    server_name = get_server_name(mt5_instance)
    out_dir = Path(out_path, server_name)

    out_dir.mkdir(parents=True, exist_ok=True)

    p = Path(out_dir / "symbol_info.json")
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Symbol info written to: {p}")

def account_info(mt5_instance: MetaTrader5, out_path: str):

    ac_info = mt5_instance.account_info()
    if ac_info is None:
        raise RuntimeError(f"account_info() failed: {mt5_instance.last_error()}")

    ac_info = ac_info._asdict() if hasattr(ac_info, "_asdict") else dict(ac_info)

    server_name = get_server_name(mt5_instance)
    out_dir = Path(out_path, server_name)

    out_dir.mkdir(parents=True, exist_ok=True)

    p = Path(out_dir / "account_info.json")

    payload = {
        "account_info": ac_info,
    }

    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Account info written to: {p}")
