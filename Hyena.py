import numpy as np
import matplotlib as plt

class Hyena():

    def __init__(self, fitnessfunction, popsize):
        self.fitnessfunction = fitnessfunction
        self.popsize = popsize