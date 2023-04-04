## 2023.02.22

# MAP3122

# using Trapezoidal method and SAM to find the solution to a bidimensional problem.

# (manufactured) problem with kwnown exact solution 
#              (1) y_1'= y_2              1<=t<=5
#              (2) y_2'= -(1/t)y_2        y_1(1) = 2; y_2(1) = 1
                         

import math
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

def implicitMethod(t0, y0, T, n):
    # compute approximate solution to the initial value problem

    y = [np.array(y0)]
    t = [t0]

    h = (T - t0) / n

    while t[-1] < T:
        # initial guess
        if(np.array(t).size > 1): 
            ytil = y[-1] + h*phi(t[-1], y[-1], t[-2], y[-2], f)
        else:
            ytil = y[-1] + h*phi(t[-1], y[-1], t[-1], y[-1], f)
        diff = 1.0

        # fixed point iteration
        r = 0
        while r<20 and diff > 0.0001:
            ytil0 = ytil
            ytil = y[-1] + h*phi(t[-1], y[-1], t[-1] + h, ytil, f)
            diff = np.linalg.norm(ytil - ytil0)
            r = r + 1
        y.append(ytil) # y(i+1) = ytil
        t.append(t[-1] + h)
    y = np.array(y)
    
    return (T - t0) / n, y[-1]

############################################################################

def ye(t):
    # exact solution 
    return np.array([np.log(t) + 2, (1/t)])

############################################################################

def main():
    # obtains the numerical convergence table based on parameters such as
    # inicial conditions, final time and number of steps

    # input numerical model data
    t0=1; y0=np.array([2, 1]);  # initial condition
    T=5             # final time
    
    # input numerical method data
    m=13;  h=[0]*m;   # number of cases to run. Initialize list of time steps
    yn=[y0]*m;       # initialize list of approximations
    
    print("MANUFACTURED SOLUTION VERIFICATION TABLE");

    # case loop
    for i in range(1,m+1): # run m times same code with h progressively small
        n=16*2**(i-1);     # number of time steps in i-th case
        
        h[i-1],yn[i-1]=implicitMethod(t0,y0,T,n);
                
        # verification via manufactured solution strategy
        # convergence table to verify the method correct implementation 
        p=q=r=0;
        
        e = max(abs(ye(T)[0]-yn[i-1][0]), abs(ye(T)[1]-yn[i-1][1]))
        if i>1:
           q = abs(max(abs(ye(T)[0]-yn[i-2][0]), abs(ye(T)[1]-yn[i-2][1]))/e)
           r = h[i-2]/h[i-1];
            
           p = math.log(q)/math.log(r);
           print("%5d & %9.3e & %9.3e & %9.3e \\\\" % (n,h[i-1],e,p))
        else: 
            print("%5d & %9.3e & %9.3e & --------- \\\\" % (n,h[i-1],e))
        
    print(" "); 

############################################################################
         
main()
