########################################################################
## Calculates the system dynamics equation f(x,u,t)
########################################################################

def kf_calc_h(t, x, u):
    
    n       = x.size
    
    # system dynamics go here
    if n == 1:
        zpred = x**3 
    elif n == 2:
        zpred = x[0]**2 + x[1]**2
        
    return zpred
        
        