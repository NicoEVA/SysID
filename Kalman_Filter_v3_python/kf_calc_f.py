########################################################################
## Calculates the system dynamics equation f(x,u,t)
########################################################################
import numpy as np

def kf_calc_f(t, x, u):
    
    n       = x.size
    xdot    = np.zeros([n,1])
    
    # system dynamics go here
    if n == 1:
        xdot[0] = -0.3*np.cos(x)**3       
    elif n == 2:
        xdot[0] = x[1]*np.cos(x[0])**3
        xdot[1] = -x[1]
        
    return xdot
        
        