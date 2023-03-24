## 2023.02.24
## Keith Ando Ogawa - keith.ando@usp.br
## Vinícius Viana de Paula - viniciusviana@usp.br

# MAP3122

# plots for general implicit one-Step methods.

# problem with unkwnown exact solution 
#              (1) x'=  0.5446x - 0.7821xy     0<=t<=5
#              (2) y'= -0.7523y + 1.7995xy     x(0) = 0.4270; y(0) = 0.6871

import matplotlib.pyplot as plt
import numpy as np

#############################################################################

def phi(t1, y1, t2, y2, f):
    # define discretization function 
    return 0.5*(f(t1, y1)+f(t2, y2))     # euler 

############################################################################

def f(t, y):
    # bidimensional problem
    f0 =  0.5446*y[0] - 0.7821*y[0]*y[1]
    f1 =  -0.7523*y[1] + 1.7995*y[0]*y[1]
    
    return np.array([f0, f1])

############################################################################

def implicitMethod(T, n, yn, tn, f):

    dt = (T - tn[-1]) / n

    while tn[-1] < T:
        # initial guess
        ytil = yn[-1] + dt*f(tn[-1], yn[-1])
        diff = 1.0

        # fixed point iteration
        r = 0
        while r<20 and diff > 0.0001:
            ytil0 = ytil
            ytil = yn[-1] + dt*phi(tn[-1], yn[-1], tn[-1] + dt, ytil, f)
            diff = np.linalg.norm(ytil - ytil0)
            r = r + 1
        yn.append(ytil) # y(i+1) = ytil
        tn.append(tn[-1] + dt)
        dt = min(dt, T-tn[-1])
    yn = np.array(yn)

    return yn, tn

############################################################################

# other relevant data
t_n_1 = [0]; t_n_2 = [0]; t_n_3 = [0]; T = 5;        # time interval: t in [t0,T]
y_n_1 = [np.array([0.427, 0.6871])]; y_n_2 = [np.array([0.427, 0.6871])];
y_n_3 = [np.array([0.427, 0.6871])]; # initial condition

n_1 = 8                # time interval partition (discretization)
y_n_1, t_n_1 = implicitMethod(T, n_1, y_n_1, t_n_1, f)

n_2 = 128                # time interval partition (discretization)
y_n_2, t_n_2 = implicitMethod(T, n_2, y_n_2, t_n_2, f)

n_3 = 256                # time interval partition (discretization)
y_n_3, t_n_3 = implicitMethod(T, n_3, y_n_3, t_n_3, f)

## plotting the graphic for x
plt.plot(t_n_1, y_n_1[:,0], 'k:', label = 'n = 8')
plt.plot(t_n_2, y_n_2[:,0], 'k--', label = 'n = 128')
plt.plot(t_n_3, y_n_3[:,0], 'k-', label = 'n = 256')


plt.xlabel('t   (em unidade de tempo)')
plt.ylabel('ω(t)  (em unidade de ω)')
plt.title('Aproximação Numérica da Variável de Estado ω')
plt.legend()
plt.show()

## plotting the graphic for y
plt.plot(t_n_1, y_n_1[:,1], 'k:' , label = 'n = 8')
plt.plot(t_n_2, y_n_2[:,1], 'k--', label = 'n = 128')
plt.plot(t_n_3, y_n_3[:,1], 'k-' , label = 'n = 256')


plt.xlabel('t   (em unidade de tempo)')
plt.ylabel('λ(t)  (em unidade de λ)')
plt.title('Aproximação Numérica da Variável de Estado λ')
plt.legend()
plt.show()

  # faz a curva em 2d
plt.plot(y_n_1[:,0], y_n_1[:,1], 'k:' , label='n = 8')
plt.plot(y_n_2[:,0], y_n_2[:,1], 'k--', label='n = 128')
plt.plot(y_n_3[:,0], y_n_3[:,1], 'k-' , label='n = 256')
plt.title('aprox para a curva em 2d')
plt.xlabel('ω (em unidade de ω)')
plt.ylabel('λ (em unidade de λ)')
plt.grid(True)
plt.legend()
plt.show()