import numpy as np
import matplotlib as plt


class Meerkat():

    def __init__(self, fitnessfunction, popsize):
        self.fitnessfunction = fitnessfunction
        self.popsize = popsize
