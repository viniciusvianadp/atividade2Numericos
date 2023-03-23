## 2023.02.22
## Keith Ando Ogawa - keith.ando@usp.br
## Vinícius Viana de Paula - viniciusviana@usp.br

# MAP3122

# plots for general implicit one-Step methods.

# (manufactured) problem with kwnown exact solution 
#              (1) y_1'= y_2              1<=t<=5
#              (2) y_2'= -(1/t)y_2        y_1(1) = 2; y_2(1) = 1

import matplotlib.pyplot as plt
import numpy as np

#############################################################################

def phi(t1, y1, t2, y2, f):
    # define discretization function 
    return 0.5*(f(t1, y1)+f(t2, y2))     # euler 

############################################################################

def f(t, y):
   # bidimensional problem
    f0 =  y[1]
    f1 =  -(1/t)*y[1]
    
    return np.array([f0, f1])

############################################################################

def implicitMethod(T, n, yn, tn, f):

    dt = (T - tn[-1]) / n

    while tn[-1] < T:
        # initial guess
        if(np.array(tn).size > 1):
            ytil = yn[-1] + dt*phi(tn[-1], yn[-1], tn[-2], yn[-2], f)
        else:
            ytil = yn[-1] + dt*phi(tn[-1], yn[-1], tn[-1], yn[-1], f)
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
t_n_1 = [1]; t_n_2 = [1]; t_n_3 = [1]; T = 5;        # time interval: t in [t0,T]
y_n_1 = [np.array([2, 1])]; y_n_2 = [np.array([2, 1])];
y_n_3 = [np.array([2, 1])]; # initial condition

n_1 = 16                # time interval partition (discretization)
y_n_1, t_n_1 = implicitMethod(T, n_1, y_n_1, t_n_1, f)

n_2 = 64                # time interval partition (discretization)
y_n_2, t_n_2 = implicitMethod(T, n_2, y_n_2, t_n_2, f)

n_3 = 128                # time interval partition (discretization)
y_n_3, t_n_3 = implicitMethod(T, n_3, y_n_3, t_n_3, f)

## plotting the graphic for y1
plt.plot(t_n_1, y_n_1[:,0], 'k:', label = 'n = 16')
plt.plot(t_n_2, y_n_2[:,0], 'k--', label = 'n = 64')
plt.plot(t_n_3, y_n_3[:,0], 'k-', label = 'n = 128')


plt.xlabel('t   (em unidade de tempo)')
plt.ylabel('y1(t)  (em unidade de y1)')
plt.title('Aproximação Numérica da Variável de Estado y1')
plt.legend()
plt.show()

## plotting the graphic for y2
plt.plot(t_n_1, y_n_1[:,1], 'k:', label = 'n = 16')
plt.plot(t_n_2, y_n_2[:,1], 'k--', label = 'n = 64')
plt.plot(t_n_3, y_n_3[:,1], 'k-', label = 'n = 128')


plt.xlabel('t   (em unidade de tempo)')
plt.ylabel('y2(t)  (em unidade de y2)')
plt.title('Aproximação Numérica da Variável de Estado y2')
plt.legend()
plt.show()

## exact vs approximated (y1)
t = np.linspace(1, 5, 65536)
plt.plot(t, np.log(t) + 2, 'k-', label = 'solução exata')
plt.plot(t_n_3, y_n_3[:, 0], 'k--', label = 'solução numérica')

plt.xlabel('t  (em unidade de tempo)')
plt.ylabel('y1(t)  (em unidade de y1)')
plt.title('Soluções Aproximada e Exata Para a Variável y1')
plt.legend()
plt.show()

## exact vs approximated (y2)
t = np.linspace(1, 5, 65536)
plt.plot(t, (1/t), 'k-', label = 'solução exata')
plt.plot(t_n_3, y_n_3[:, 1], 'k--', label = 'solução numérica')

plt.xlabel('t  (em unidade de tempo)')
plt.ylabel('y2(t)  (em unidade de y2)')
plt.title('Soluções Aproximada e Exata Para a Variável y2')
plt.legend()
plt.show()
