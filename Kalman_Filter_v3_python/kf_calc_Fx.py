########################################################################
## Calculates the Jacobian of the system dynamics equation 
########################################################################
import numpy as np

def kf_calc_Fx(t, x, u):
    
    n = x.size
    
    # calculate Jacobian matrix of system dynamics
    if (n == 1):
        DFx = 0.9*np.cos(x)**2 * np.sin(x)
    elif (n == 2):
        DFx         = np.zeros([2, 2]);
        DFx[0, 0]   = x[1]*np.cos(x[0])**2 * np.sin(x[0])
        DFx[0, 1]   = np.cos(x[0])**3
        DFx[1, 0]   = 0
        DFx[1, 1]   = -1
        
    return DFx
        