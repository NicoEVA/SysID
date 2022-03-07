from scipy.io import loadmat
import math
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from os import listdir

def get_maneuvers(path:str):
    maneuvers = listdir(path)
    return maneuvers

def load_speeds(path):
    data = loadmat(path)
    u_n = data["u_n"]
    v_n = data["v_n"]
    w_n = data["w_n"]
    Theta = np.deg2rad(data["theta"])
    Phi = np.deg2rad(data["phi"])
    Psi = np.deg2rad(data["psi"])
    timescale = data["t"]

    return u_n, v_n, w_n, Theta, Phi, Psi, timescale


def convert_speed_to_position(path):
    x_path = []
    y_path = []
    z_path = []
    u_n, v_n, w_n, Theta, Phi, Psi, timescale = load_speeds(path)
    dt = timescale[2]-timescale[1]
    WxE = -10
    WyE = 3
    WzE = 2
    x = 0
    y = 0
    z = 0
    for i in range(len(u_n)):
        # dx = (u_n[i]*math.cos(Theta[i])+(v_n[i]*math.sin(Phi[i])+w_n[i]*math.cos(Phi[i]))*math.sin(Theta[i]))*math.cos(Psi[i])-(v_n[i]*math.cos(Phi[i])-w_n[i]*math.sin(Phi[i]))*math.sin(Psi[i])+WxE
        # dy = (u_n[i]*math.cos(Theta[i])+(v_n[i]*math.sin(Phi[i])+w_n[i]*math.cos(Phi[i]))*math.sin(Theta[i]))*math.sin(Psi[i])+(v_n[i]*math.cos(Phi[i])-w_n[i]*math.sin(Phi[i]))*math.cos(Psi[i])+WyE
        # dz = -u_n[i]*math.sin(Theta[i])+(v_n[i]*math.sin(Phi[i])+w_n[i]*math.cos(Phi[i]))*math.cos(Theta[i]) + WzE

        x += (u_n[i][0] + WxE) * 0.01
        y += (v_n[i][0] + WyE) * 0.01
        z += (w_n[i][0] + WzE) * 0.01
        x_path.append(x)
        y_path.append(y)
        z_path.append(z)

    return x_path, y_path, z_path

def plot_path_and_speeds(path):
    x_path, y_path, z_path = convert_speed_to_position(path)
    fig = plt.figure()

    # syntax for 3-D projection
    ax = plt.axes(projection='3d')

    # defining axes
    x = x_path
    y = y_path
    z = z_path

    color_map = plt.get_cmap('spring')
    ax.scatter(x[:2000], y[:2000], z[:2000], cmap=color_map)

    # syntax for plotting
    ax.set_title('3d Scatter plot geeks for geeks')
    plt.show()





