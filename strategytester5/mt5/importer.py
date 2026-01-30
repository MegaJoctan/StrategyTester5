from strategytester5 import AccountInfo, SymbolInfo
import json
import os
import logging

def account_info(broker_path: str, logger: logging.Logger = None) -> AccountInfo:

    try:
        file = os.path.join(broker_path, "account_info.json")
    except Exception as e:
        err = "Invalid broker path"
        if logger is None:
            print(err)
        else:
            logger.error(err)
        return dict()

    if not os.path.exists(file):
        err = f"Failed to import account info, {file} not found"
        if logger is None:
            print(err)
        else:
            logger.error(err)

        return dict()

    with open(file) as json_file:
        data = json.load(json_file)

    account_info = AccountInfo(**data["account_info"])
    return account_info

def all_symbol_info(broker_path: str, logger: logging.Logger = None) -> tuple:

    try:
        file = os.path.join(broker_path, "symbol_info.json")
    except Exception as e:
        err = "Invalid broker path"
        if logger is None:
            print(err)
        else:
            logger.error(err)
        return dict()

    if not os.path.exists(file):
        err = f"Failed to import symbol_info, {file} not found"
        if logger is None:
            print(err)
        else:
            logger.error(err)
        return tuple()

    with open(file) as json_file:
        data = json.load(json_file)

    all_symbol_info = []

    for s in data["symbols"]:
        all_symbol_info.append(SymbolInfo(**s))

    return tuple(all_symbol_info)