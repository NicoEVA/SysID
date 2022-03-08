from scipy.io import loadmat
import math
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

def load_data(path):
    data = loadmat(path)
    return data


