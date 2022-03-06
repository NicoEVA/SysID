########################################################################
## Calculates the Jacobian of the output dynamics equation 
########################################################################
import numpy as np

def kf_calc_Hx(t, x, u):
    
    n = x.size
    
    # calculate Jacobian matrix of system dynamics
    if (n == 1):
        Hx = 3*x**2
    elif (n == 2):
        Hx          = np.zeros([1, 2]);
        Hx[0, 0]    = 2*x[0]
        Hx[0, 1]    = 2*x[1]
        
    return Hx
        