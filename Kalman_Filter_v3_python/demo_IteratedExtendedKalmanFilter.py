import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import control.matlab
from kf_calc_f import kf_calc_f
from kf_calc_h import kf_calc_h
from rk4 import rk4
from kf_calc_Fx import kf_calc_Fx
from kf_calc_Hx import kf_calc_Hx
plt.close('all')
np.random.seed(7)
pathname = os.path.basename(sys.argv[0])
filename = os.path.splitext(pathname)[0]

########################################################################
## Set simulation parameters
########################################################################

n               = 1                 # state dimension
nm              = 1                 # number of measurements
m               = 1                 # number of inputs
dt              = 0.01              # time step (s)
N               = 1000              # sample dimension
epsilon         = 10**(-10)         # IEKF threshold
doIEKF          = True              # If false, EKF without iterations is used
maxIterations   = 100               # maximum amount of iterations per sample

printfigs       = False             # enable saving figures
figpath         = ''                # direction for printed figures

########################################################################
## Set initial values for states and statistics
########################################################################

E_x_0       = 10                # initial estimate of optimal value of x_k1_k1
x_0         = 5                 # initial true state

B           = 1                 # input matrix
G           = 1                 # noise input matrix

# Initial estimate for covariance matrix
std_x_0     = 10                # initial standard deviation of state prediction error
P_0         = std_x_0**2        # initial covariance of state prediction error

# System noise statistics
E_w         = 0                 # bias of system noise
std_w       = 1                 # standard deviation of system noise
Q           = std_w**2          # variance of system noise
w_k         = std_w*np.random.normal(0, 1, [n, N]) + E_w    # system noise

# Measurement noise statistics
E_v         = 0                 # bias of measurement noise
std_v       = 5                 # standard deviation of measurement noise
R           = std_v**2          # variance of measurement noise
v_k         = std_v*np.random.normal(0, 1, [n, N]) + E_v    # measurement noise

########################################################################
## Generate batch with measurement data
########################################################################

x           = np.array(x_0)     # initial true state
X_k         = np.zeros([n, N])  # true state
Z_k         = np.zeros([nm, N]) # measurement
U_k         = np.zeros([m, N])  # inputs

for i in range(0, N):
    dx          = kf_calc_f(0, x, U_k[:,i])             # noiseless state derivative
    x           = x + (dx + w_k[:,i])*dt                # true state
    z           = kf_calc_h(0, x, U_k[:,i]) + v_k[:,i]  # measurement                            # measurement
    X_k[:,i]    = x                                     # store true state
    Z_k[:,i]    = z                                     # store measurement  

########################################################################
## Initialize Extended Kalman filter
########################################################################

t_k         = 0
t_k1        = dt

# allocate space to store traces
XX_k1_k1    = np.zeros([n, N])
PP_k1_k1    = np.zeros([n, N])
STD_x_cor   = np.zeros([n, N])
STD_z       = np.zeros([nm, N])
ZZ_pred     = np.zeros([nm, N])
IEKFitcount = np.zeros([N, 1])

# initialize state estimation and error covariance matrix
x_k1_k1     = np.array(E_x_0)   # x(0|0) = E(x_0)
P_k1_k1     = np.array(P_0)     # P(0|0) = P(0)

########################################################################
## Run the Kalman filter
########################################################################

t0          = time.time()

# Run the filter through all N samples
for k in range(0, N):
    
    # x(k+1|k) (prediction)
    t, x_k1_k   = rk4(kf_calc_f, x_k1_k1, U_k[:,k], [t_k, t_k1])   

    # Calc Jacobians Phi(k+1, k) and Gamma(k+1, k)
    Fx          = kf_calc_Fx(0, x_k1_k, U_k[:,k])
    # Continuous to discrete time transformation of Df(x,u,t)
    ss_B        = control.matlab.ss(Fx, B, 0, 0)
    ss_G        = control.matlab.ss(Fx, G, 0, 0)
    Psi         = control.matlab.c2d(ss_B, dt).B
    Phi         = control.matlab.c2d(ss_G, dt).A
    Gamma       = control.matlab.c2d(ss_G, dt).B

    # P(k+1|k) (prediction covariance matrix)
    P_k1_k      = Phi*P_k1_k1*Phi.transpose() + Gamma*Q*Gamma.transpose()
    
    # Run the Iterated Extended Kalman filter (if doIEKF = 1), else run standard EKF
    if (doIEKF == True):
        
        eta2    = x_k1_k
        err     = 2*epsilon
        itts    = 0
        
        while (err > epsilon):
            if (itts >= maxIterations):
                print("Terminating IEKF: exceeded max iterations (%d)\n" %(maxIterations))  
                break
            
            itts    = itts + 1
            eta1    = eta2
              
            # Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix 
            Hx           = kf_calc_Hx(0, eta1, U_k[:,k])
            
            # Observation and observation error predictions
            z_k1_k      = kf_calc_h(0, eta1, U_k[:,k])                         # prediction of observation (for validation)   
            P_zz        = Hx*P_k1_k*Hx.transpose() + R      # covariance matrix of observation error (for validation)   
            std_z       = np.sqrt(np.diagflat(P_zz))           # standard deviation of observation error (for validation)    
    
            # K(k+1) (gain)
            K           = P_k1_k*Hx.transpose()/P_zz   
        
            # new observation
            eta2        = x_k1_k + K*(Z_k[:,k] - z_k1_k - Hx*(x_k1_k - eta1))
            err         = np.linalg.norm(eta2-eta1)/np.linalg.norm(eta1)  
    
        IEKFitcount[k]  = itts
        x_k1_k1         = eta2
    
    else:
        # Correction
        Hx          = kf_calc_Hx(0, x_k1_k, U_k[:,k])        

        # P_zz(k+1|k) (covariance matrix of innovation)
        z_k1_k      = kf_calc_h(0, x_k1_k, U_k[:,k])        
        P_zz        = Hx*P_k1_k*Hx.transpose() + R      # covariance matrix of observation error (for validation)   
        std_z       = np.sqrt(np.diagflat(P_zz))           # standard deviation of observation error (for validation)    
    
        # K(k+1) (gain)
        K           = P_k1_k*Hx.transpose()/P_zz    
        
        # Calculate optimal state x(k+1|k+1) 
        x_k1_k1     = x_k1_k + K*(Z_k[:,k] - z_k1_k)
       
    # P(k|k) (correction) using the numerically stable form of P_k_1k_1 = (eye(n) - K*Hx) * P_kk_1 
    P_k1_k1     = (np.eye(n) - K*Hx)*P_k1_k*(np.eye(n) - K*Hx).transpose() + K*R*K.transpose()    
    std_x_cor   = np.sqrt(np.diagflat(P_k1_k1))        # standard deviation of state estimation error (for validation)

    # Next step
    t_k         = t_k1 
    t_k1        = t_k1 + dt
    
    # store results
    ZZ_pred[:,k]    = z_k1_k    
    XX_k1_k1[:,k]   = x_k1_k1
    PP_k1_k1[:,k]   = P_k1_k1
    STD_x_cor[:,k]  = std_x_cor
    STD_z[:,k]      = std_z

t1              = time.time()

# calculate state estimation error (in real life this is unknown!)
EstErr_x    = XX_k1_k1-X_k

# calculate measurement estimation error (possible in real life)
EstErr_z    = ZZ_pred-Z_k

print("IEKF state estimation error RMS = %.3E, completed run with %d samples in %.2f seconds." %(np.sqrt((np.square(EstErr_x)).mean()), N, t1-t0))  

########################################################################
## Plotting
########################################################################
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 1 ,1 )
ax.plot(X_k.transpose(), 'b')
ax.plot(XX_k1_k1.transpose(), 'r')
ax.plot(Z_k.transpose(), 'k')
plt.xlim(0,N)
plt.grid(True)
plt.title('True State, estimated state and measurement')
plt.legend(['true state', 'estimated state', 'measurement'], loc='upper right')
plt.show()
if printfigs == True:
    fig.savefig(filename+'_TrueStateEstimatedStateMeasurement.png', bbox_inches='tight')

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 1 ,1 )
ax.plot(EstErr_x.transpose(), 'b')
ax.plot(STD_x_cor.transpose(), 'r')
ax.plot(-STD_x_cor.transpose(), 'g')
plt.xlim(0,N)
plt.ylim(np.min(EstErr_x), np.max(EstErr_x))
plt.grid(True)
plt.title('State estimation error with STD')
plt.legend(['estimation error', 'upper error STD', 'lower error STD'], loc='upper right')
plt.show()
if printfigs == True:
    fig.savefig(filename+'_StateEstimationError.png', bbox_inches='tight')

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 1 ,1 )
ax.plot(EstErr_z.transpose(), 'b')
ax.plot(STD_z.transpose(), 'r')
ax.plot(-STD_z.transpose(), 'g')
plt.xlim(0,N)
plt.ylim(np.min(EstErr_z), np.max(EstErr_z))
plt.grid(True)
plt.title('Measurement estimation error with STD')
plt.legend(['estimation error', 'upper error STD', 'lower error STD'], loc='upper right')
plt.show()
if printfigs == True:
    fig.savefig(filename+'_MeasurementEstimationError.png', bbox_inches='tight')
    
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 1 ,1 )
ax.plot(IEKFitcount, 'b')
plt.xlim(0, N)
if (np.max(IEKFitcount) > 0):
    plt.ylim(0, np.max(IEKFitcount))
plt.grid(True)
plt.title('IEKF iterations at each sample')
plt.show()
if printfigs == True:
    fig.savefig(filename+'_NumberOfIterations.png', bbox_inches='tight')