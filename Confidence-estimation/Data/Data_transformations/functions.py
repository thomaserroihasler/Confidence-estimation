import math as mt

def typical_displacement(Temperature,cut_off, n):
    if isinstance(cut_off, (float, int)):
        log = mt.log(cut_off)
    else:
        log = cut_off.log()
    return n * (mt.pi * Temperature * log) ** .5 / 2