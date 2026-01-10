import json
import time
from enum import Enum

from json.decoder import JSONDecodeError
from datetime import datetime

class FaultType(Enum):
    Raise = 0
    Timeout = 1

    def __str__(self):
        return self._get_fault_type_string()

    def _get_fault_type_string(self):
        fault_type_mapping = {
            FaultType.Raise: "raise",
            FaultType.Timeout: "timeout",
        }
        return fault_type_mapping.get(self, self.name.lower())

def check_raise(file_path='./recipe/fault_recover/fault_flag.json', fault_type:str="raise", timeout=60):
    while True:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_raise_flag = json.load(f)['fault_flag']
        except JSONDecodeError:
            time.sleep(1)
        else:
            break
    if file_raise_flag:
        if fault_type == str(FaultType.Raise):
            raise Exception(f"[fault_manager][{datetime.now()}] raise exception with check_raise util")
        if fault_type == str(FaultType.Timeout):
            print(f"[fault_manager][{datetime.now()}] set timeout {timeout} with check_raise util")
            time.sleep(timeout)
            print(f"[fault_manager][{datetime.now()}] end timeout {timeout} with check_raise util")