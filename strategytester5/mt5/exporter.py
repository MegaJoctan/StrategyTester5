import os
import json


def all_symbol_info(mt5_instance: any, out_path: str):
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
            skipped.append({
                "symbol": name,
                "reason": "symbol_info returned None"
            })
            continue

        # SymbolInfo is namedtuple-like
        d = info._asdict() if hasattr(info, "_asdict") else dict(info)
        exported.append(d)

    payload = {
        "exported": len(exported),
        "skipped": skipped,
        "symbols": exported,
    }

    # Create directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)
    file_path = os.path.join(out_path, "symbol_info.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Symbol info written to: {file_path}")


def account_info(mt5_instance: any, out_path: str):
    ac_info = mt5_instance.account_info()
    if ac_info is None:
        raise RuntimeError(f"account_info() failed: {mt5_instance.last_error()}")

    ac_info = ac_info._asdict() if hasattr(ac_info, "_asdict") else dict(ac_info)

    # Ensure directory exists
    os.makedirs(out_path, exist_ok=True)
    file_path = os.path.join(out_path, "account_info.json")

    payload = {
        "account_info": ac_info,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Account info written to: {file_path}")
