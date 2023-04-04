## 2023.03.24

# MAP3122

# using Trapezoidal method and SAM to find the solution to a bidimensional problem.

# problem with unkwnown exact solution 
#              (1) x'=  0.4716x - 0.6696xy     0<=t<=5
#              (2) y'= -0.2154y + 0.5240xy     x(0) = 0.4270; y(0) = 0.6871
                         

import math
import numpy as np

#############################################################################

def phi(t1, y1, t2, y2, f):
    # define discretization function 
    return 0.5*(f(t1, y1)+f(t2, y2)) 

############################################################################

def f(t, y):
    # bidimensional problem
    f0 =  0.4716*y[0] - 0.6696*y[0]*y[1]
    f1 =  -0.2154*y[1] + 0.5240*y[0]*y[1]
    
    return np.array([f0, f1])

############################################################################

def implicitMethod(t0, y0, T, n):
    # compute approximate solution to the initial value problem

    y = [np.array(y0)]
    t = [t0]

    h = (T - t0) / n

    while t[-1] < T:
        # initial guess
        ytil = y[-1] + h*f(t[-1], y[-1])
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

def main():
    # obtains the numerical convergence table based on parameters such as
    # inicial conditions, final time and number of steps

    # input numerical model data
    t0=0; y0=np.array([0.427, 0.6871]);  # initial condition
    T=5             # final time
    
    # input numerical method data
    m=13;  h=[0]*m;   # number of cases to run. Initialize list of time steps
    yn=[y0]*m;       # initialize list of approximations
    
    with open("behavior_convergence.txt", 'w', encoding='utf-8') as file2:
        file2.write("ORDER BEHAVIOR CONVERGENCE TABLE\n");
    
        e=p=q=r=s1=s2=0;
        for i in range(1,m+1):
            n=16*2**(i-1); 

            h[i-1],yn[i-1]=implicitMethod(t0,y0,T,n);
            if i>2:
                z3= np.array(yn[i-3])
                z2 = np.array(yn[i-2])
                z1 = np.array(yn[i-1])
                s1 = np.sqrt(((z3 - z2)[0])**2 +((z3 - z2)[1])**2)
                s2 = np.sqrt(((z2 - z1)[0])**2 +((z2 - z1)[1])**2)

                q = s1/s2;
                r = h[i-2]/h[i-1];
            
                p = math.log(q)/math.log(r);
            
                e = ((z2-z1)[0]**2 + (z2-z1)[1]**2)/3; 
                print("%7d & %9.3e & %9.3e & %9.3e \\\\" % (n,h[i-1],e,p))
                file2.write("{:5d} & {:9.3e} & {:9.3e} & {:9.3e}\\\\\n".format(n,h[i-1],e,p)) 

############################################################################
         
main()
