import os
from typing import Optional, Union, Dict, Any

import numpy as np
from neptune.new import Run


class Log:
    def __init__(self):
        self._memory = []
        self._item = None

    def log(self, value):
        self._memory.append(value)

    def upload(self, item):
        self._item = item

    def get_series(self):
        if self._memory:
            return self._memory
        else:
            return self._item


class Logger:

    def __init__(self):
        self.dict = {}

    def stop(self, seconds: Optional[Union[float, int]] = None):
        self.save()

    def fetch(self) -> dict:
        raise NotImplementedError

    def assign(self, value, wait: bool = False) -> None:
        raise NotImplementedError

    def get_structure(self) -> Dict[str, Any]:
        raise NotImplementedError

    def print_structure(self) -> None:
        raise NotImplementedError

    def pop(self, path: str, wait: bool = False) -> None:
        raise NotImplementedError

    def wait(self, disk_only=False) -> None:
        raise NotImplementedError

    def sync(self, wait: bool = True) -> None:
        raise NotImplementedError

    def __getitem__(self, item):
        if item not in self.dict:
            self.dict[item] = Log()
        return self.dict[item]

    def __setitem__(self, key: str, value) -> None:
        self.dict[key] = value

    def __delitem__(self, path) -> None:
        raise NotImplementedError

    def fetch(self) -> dict:
        raise NotImplementedError

    def save(self):
        import torch
        from datetime import datetime
        dts = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(self, f"logs/saved_loggers/{self.dict['skein_id']}_{dts}.lgr")


def main():
    valid_paths = []
    base_dir = "logs/saved_loggers"
    for path in os.listdir(base_dir):
        if path[-4:] == ".lgr":
            valid_paths.append(os.path.join(base_dir, path))
    print(valid_paths)
    import enquiries
    choice = enquiries.choose('Choose something to load: ', valid_paths)
    print(f"CHOSEN: {choice}")
    import torch
    lgr = torch.load(choice)
    lg_dict = lgr.dict

    for key in lg_dict.keys():
        if "gif" not in key:
            print(key, "    ", lg_dict[key])

    keys = ["ep_score", "dist_score", "spec_score"]

    def smooth(xs: list, resolution: int = 100) -> list:
        xs = np.array(xs)
        xs = np.mean(xs.reshape(-1, resolution), axis=1)
        return xs

    import matplotlib.pyplot as plt
    for key in keys:
        log_item = lg_dict[key]
        plt.plot(smooth(log_item.get_series()))
    plt.show()


if __name__ == "__main__":
    main()
