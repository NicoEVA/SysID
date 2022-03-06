import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import control.matlab
from kf_calc_lin_F import kf_calc_lin_F
from kf_calc_lin_H import kf_calc_lin_H
plt.close('all')
np.random.seed(7)
pathname = os.path.basename(sys.argv[0])
filename = os.path.splitext(pathname)[0]

########################################################################
## Set simulation parameters
########################################################################

n           = 1                 # state dimension
dt          = 0.01              # time step (s)
N           = 1000              # sample dimension

printfigs   = False             # enable saving figures
figpath     = ''                # direction for printed figures

########################################################################
## Set initial values for states and statistics
########################################################################

E_x_0       = 10                # initial estimate of optimal value of x_k1_k1
x_0         = 1                 # initial true state
u           = 0                 # system input
B           = 0                 # input matrix
G           = 1                 # noise input matrix

# Initial estimate for covariance matrix
std_x_0     = 10                # initial standard deviation of state prediction error
P_0         = std_x_0**2        # initial covariance of state prediction error

# System noise statistics
E_w         = 0                 # bias of system noise
std_w       = 100               # standard deviation of system noise
Q           = std_w**2          # variance of system noise
w_k         = std_w*np.random.normal(0, 1, [N, n]) + E_w    # system noise

# Measurement noise statistics
E_v         = 0                 # bias of measurement noise
std_v       = 2                 # standard deviation of measurement noise
R           = std_v**2          # variance of measurement noise
v_k         = std_v*np.random.normal(0, 1, [N, n]) + E_v    # measurement noise

########################################################################
## Generate batch with measurement data
########################################################################

x           = x_0               # initial true state
X_k         = np.zeros([N, 1])  # true state
Z_k         = np.zeros([N, 1])  # measurement
U_k         = np.zeros([N, 1])  # inputs

for i in range(0, N):
    dx          = kf_calc_lin_F(0, x, U_k[i])           # noiseless state derivative
    x           = x + (dx + w_k[i,:])*dt                # true state
    z           = kf_calc_lin_H(0, x, U_k[i]) + v_k[i]  # measurement                            # measurement
    X_k[i,:]    = x                                     # store true state
    Z_k[i,:]    = z                                     # store measurement 
    
########################################################################
## Initialize Kalman filter
########################################################################

t_k         = 0
t_k1        = dt

# allocate space to store traces
XX_k1_k1    = np.zeros([N, n])
PP_k1_k1    = np.zeros([N, n])
STD_x_cor   = np.zeros([N, n])
STD_z       = np.zeros([N, n])
ZZ_pred     = np.zeros([N, n])

# initialize state estimation and error covariance matrix
x_k1_k1     = E_x_0             # x(0|0) = E(x_0)
P_k1_k1     = P_0               # P(0|0) = P(0)

# calculate discrete state transition matrix and prediction (time and state invariant)
F           = np.array(kf_calc_lin_F(0, 1, u))
ss_B        = control.matlab.ss(F, B, 0, 0)
ss_G        = control.matlab.ss(F, G, 0, 0)
Psi         = control.matlab.c2d(ss_B, dt).B
Phi         = control.matlab.c2d(ss_G, dt).A
Gamma       = control.matlab.c2d(ss_G, dt).B

# calculate discrete state observation matrix (time invariant)
H           = np.array(kf_calc_lin_H(0, 1, 0))

########################################################################
## Run the Kalman filter
########################################################################
t0          = time.time()

# Run the filter through all N samples
for k in range(0, N):
    # x(k+1|k) (prediction)
    x_k1_k      = Phi*x_k1_k1 + Psi*U_k[k]

    # P(k+1|k) (prediction covariance matrix)
    P_k1_k      = Phi*P_k1_k1*Phi.transpose() + Gamma*Q*Gamma.transpose()
    
    # Observation and observation error predictions
    z_k1_k      = H*x_k1_k                         # prediction of observation (for validation)   
    P_zz        = H*P_k1_k*H.transpose() + R       # covariance matrix of observation error (for validation)   
    std_z       = np.sqrt(np.diagflat(P_zz))       # standard deviation of observation error (for validation)    

    # K(k+1) (gain)
    K           = P_k1_k*H.transpose()/(H*P_k1_k*H.transpose() + R)   
    
    # Calculate optimal state x(k+1|k+1) 
    x_k1_k1     = x_k1_k + K*(Z_k[k] - H*x_k1_k)         

    # P(k|k) (correction)
    P_k1_k1     = (np.eye(n) - K*H)*P_k1_k    
    std_x_cor   = np.sqrt(np.diagflat(P_k1_k1))        # standard deviation of state estimation error (for validation)

    # Next step
    t_k         = t_k1 
    t_k1        = t_k1 + dt
    
    # store results
    XX_k1_k1[k]     = x_k1_k1
    PP_k1_k1[k]     = P_k1_k1
    STD_x_cor[k]    = std_x_cor
    STD_z[k]        = std_z
    ZZ_pred[k]      = z_k1_k        

t1              = time.time()

# calculate state estimation error (in real life this is unknown!)
EstErr_x    = XX_k1_k1-X_k

# calculate measurement estimation error (possible in real life)
EstErr_z    = ZZ_pred-Z_k

print("KF state estimation error RMS = %.3E, completed run with %d samples in %.2f seconds." %(np.sqrt((np.square(EstErr_x)).mean()), N, t1-t0)) 

########################################################################
## Plotting
########################################################################
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 1 ,1 )
ax.plot(X_k, 'b')
ax.plot(XX_k1_k1, 'r')
ax.plot(Z_k, 'k')
plt.xlim(0,N)
plt.grid(True)
plt.title('True State, estimated state and measurement')
plt.legend(['true state', 'estimated state', 'measurement'], loc='upper right')
plt.show()
if printfigs == True:
    fig.savefig(filename+'_TrueStateEstimatedStateMeasurement.png', bbox_inches='tight')

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 1 ,1 )
ax.plot(EstErr_x, 'b')
ax.plot(STD_x_cor, 'r')
ax.plot(-STD_x_cor, 'g')
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
ax.plot(EstErr_z, 'b')
ax.plot(STD_z, 'r')
ax.plot(-STD_z, 'g')
plt.xlim(0,N)
plt.ylim(np.min(EstErr_z), np.max(EstErr_z))
plt.grid(True)
plt.title('Measurement estimation error with STD')
plt.legend(['estimation error', 'upper error STD', 'lower error STD'], loc='upper right')
plt.show()
if printfigs == True:
    fig.savefig(filename+'_MeasurementEstimationError.png', bbox_inches='tight')