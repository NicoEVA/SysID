from scipy.io import loadmat
import math
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

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

        x += (u_n[i] + WxE) * dt
        y *= (v_n[i] + WyE) * dt
        z += (w_n[i] + WzE) * dt
        x_path.append(x)
        y_path.append(y)
        z_path.append(z)

    return x_path, y_path, z_path


u_n, v_n, w_n, Theta, Phi, Psi, timescale = load_speeds("./simdata2021/de3211.mat")

WxE = -10
WzE = 2
WyE = 3
x = 0
h = 0
d = 0
x_path = []
h_path = []
d_path = []
for i in range(len(u_n)):
    # dx = (u_n[i]*math.cos(Theta[i])+(v_n[i]*math.sin(Phi[i])+w_n[i]*math.cos(Phi[i]))*math.sin(Theta[i]))*math.cos(Psi[i])-(v_n[i]*math.cos(Phi[i])-w_n[i]*math.sin(Phi[i]))*math.sin(Psi[i])+WxE
    # dy = (u_n[i]*math.cos(Theta[i])+(v_n[i]*math.sin(Phi[i])+w_n[i]*math.cos(Phi[i]))*math.sin(Theta[i]))*math.sin(Psi[i])+(v_n[i]*math.cos(Phi[i])-w_n[i]*math.sin(Phi[i]))*math.cos(Psi[i])+WyE
    # dz = -u_n[i]*math.sin(Theta[i])+(v_n[i]*math.sin(Phi[i])+w_n[i]*math.cos(Phi[i]))*math.cos(Theta[i]) + WzE

    x += (u_n[i][0] + WxE) * 0.01
    h += (w_n[i][0] + WzE) * 0.01
    d += (v_n[i][0] + WyE) * 0.01
    x_path.append(x)
    h_path.append(h)
    d_path.append(d)

plt.figure()
plt.plot(u_n)
plt.plot(w_n)

plt.figure()
plt.plot(h_path)

plt.figure()
plt.plot(x_path)

plt.figure()
plt.plot(d_path)

plt.show()

fig = plt.figure()

# syntax for 3-D projection
ax = plt.axes(projection='3d')

# defining axes
z = h_path
x = x_path
y = d_path
color_map = plt.get_cmap('spring')
ax.scatter(x[:2000], y[:2000], z[:2000], cmap=color_map)

# syntax for plotting
ax.set_title('3d Scatter plot geeks for geeks')
plt.show()

