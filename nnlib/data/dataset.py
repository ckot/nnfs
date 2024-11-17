from functools import cache
from typing import Tuple

import numpy as np

class DataSet:

    def __len__(self):
        raise NotImplementedError

    @cache
    def __getitem__(self, index: int) -> Tuple[np.array, np.int64]:
        raise NotImplementedError

