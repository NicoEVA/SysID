import reader_functions
from reader import load_data
from reader_functions import *
import numpy as np

path = "./simdata2021"

maneuvers = reader_functions.get_maneuvers(path)
print(maneuvers)


data_dict = {}
for maneuver in maneuvers:
    data = load_data(f"{path}/{maneuver}")
    data_dict.update({maneuver: data})



def gen_GPS(data):
    for i in range(len(data["t"])):
        u_n     = data["u_n"]
        v_n     = data["v_n"]
        w_n     = data["w_n"]
        theta = data["theta"]
        phi   = data["phi"]
        psi   = data["psi"]
        WxE = -10
        WyE = 3
        WzE = 2

        # u_gps = (u_n[i]*math.cos(theta[i])+(v_n[i]*math.sin(phi[i])+w_n[i]*math.cos(phi[i]))*math.sin(theta[i]))*math.cos(psi[i])-(v_n[i]*math.cos(phi[i])-w_n[i]*math.sin(phi[i]))*math.sin(psi[i])+WxE
        # v_gps = (u_n[i]*math.cos(theta[i])+(v_n[i]*math.sin(phi[i])+w_n[i]*math.cos(phi[i]))*math.sin(theta[i]))*math.sin(psi[i])+(v_n[i]*math.cos(phi[i])-w_n[i]*math.sin(phi[i]))*math.cos(psi[i])+WyE
        # w_gps = -u_n[i]*math.sin(theta[i])+(v_n[i]*math.sin(phi[i])+w_n[i]*math.cos(phi[i]))*math.cos(theta[i]) + WzE
        #all other data seems to stay as is

def gen_IMU(data):
