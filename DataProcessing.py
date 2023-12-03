import torch as tr
import math as mt

class ConvertLabelsToInt:
    def __call__(self, label):
        return tr.tensor(label, dtype=tr.int)

