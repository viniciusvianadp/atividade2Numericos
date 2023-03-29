## 2023.03.29
## Vinícius Viana de Paula - viniciusviana@usp.br

# MAP3122

## Implementation of the Least Square Method
## The program will receive the values of the percentage share of gross
## domestic income and employment rate throughout the studied years, and
## their respective derivatives are obtained using a polynomial interpolator.
## Then, the equations for the lines are obtained, using the least square method.

## The Goodwin model equations are approximated using the following expressions:
## ŷ_1 = (b_1)x_1 + b_0 <=> ω'/ω = aλ + b; 
## ŷ_2 = (c_1)x_2 + c_0 <=> λ'/λ = cω + d.

import numpy as np
import matplotlib.pyplot as plt

##########################################################
## Obtains the polynomial interpolator
##########################################################
def polynomial(x, y):
    n = len(x) - 1
    A = []
    for xi in x:
        row = [1]
        for j in range(1, n + 1):
            row.append(xi ** j)
        A.append(row)
    return np.linalg.solve(A, y)

##########################################################
## Receives the coef matrix and the point of interest and
## returns the derivative at the given point.
##########################################################
def derivative(A, xi):
    n = len(A) - 1
    dv = 0
    for j in range(1, n + 1):
        dv = dv + j*A[j]*(xi**(j-1)) 
    return dv

##########################################################
## Retuns the value of the function for the given point x
##########################################################
def func_poly(x, coeffs):
    first = coeffs[0]
    return first + sum([ai * x ** j for j, ai in enumerate(coeffs[1:], 1)])

##########################################################
## Returns the coefs for the obtained equation using the lsm
##########################################################
def least_square_method(y, x):
    x_sqr = []
    xy = []

    for i in range(len(x)):
        x_sqr.append(x[i] * x[i])
        xy.append(x[i]*y[i])

    sum_xy = sum([num for num in xy])
    sum_x = sum([num for num in x])
    sum_y = sum([num for num in y])
    sum_x_sqr = sum([num for num in x_sqr])

    s_xy = sum_xy - (sum_x * sum_y)/len(y)
    s_xx = sum_x_sqr - (sum_x * sum_x)/len(y)
    
    a = s_xy / s_xx
    b = (sum_y)/len(y) - a * (sum_x)/len(y)    

    print( "              y         x       x^2        xy")
    for i in range(len(y)):
        print(f"      {y[i]:.3e} {x[i]:.3e} {x_sqr[i]:.3e} {xy[i]:.3e}")
    print(f"Total {sum_y:.3e} {sum_x:.3e} {sum_x_sqr:.3e} {sum_xy:.3e}")
    print(f"Media {(sum_y/len(y)):.3e} {(sum_x/len(y)):.3e}")    
    print("Resultados")
    print(f"Sxx  {s_xx:.3e}  Sxy  {s_xy:.3e}")
    print(f"a   {a:.4e} b   {b:.4e}")

    return a, b # y = ax + b 

##########################################################
## Prints the information related to the linear regression
## study
##########################################################
def linear_regression(a, b, y, x):
     y_hat = []
     y_med = sum(y)/len(y) 

     for xi in x:
         y_hat.append(a*xi + b)
    
     # (yi - ymed)^2 -> total sum
     TS = sum([(yi-y_med)**2 for yi in y])
     # (yi - yhat)^2 -> residual sum
     RS = sum([(y[i]-y_hat[i])**2 for i in range(len(y))])
     # regression sum (TS - RS)
     REG = TS - RS
     
     # Degrees of freedom
     GLT = len(y) - 1
     GLR = len(y) - 2
     # Snedecor F
     F = REG / (RS/GLR)

     print("                  SQ  GL        QM       F")
     print(f"Regressao  {REG:1.3e}  1  {REG:1.3e}  {F:1.3f}")
     print(f"Residual   {RS:1.3e}  {GLR}  {(RS/GLR):1.3e}")
     print(f"Total      {TS:1.3e}  {GLT}  {(TS/GLT):1.3e}")
##########################################################
##########################################################

##########################################################
## Usage of the given problem's data
##########################################################

# data
years = [0, 1, 2]
gross_income = [0.4270, 0.4310, 0.4330]
employment_rate = [0.6871, 0.6935, 0.7011]
gross_deriv = []
employ_deriv = []
y1 = []
y2 = []

# obtains ω' for each year
gross_coeffs = polynomial(years, gross_income)
for i in range(len(gross_income)):
    gross_deriv.append(derivative(gross_coeffs, i))

print("Gross Income")
for i in range(len(gross_income)):
    print(f"Value: {gross_income[i]:1.3e} Derivative: {gross_deriv[i]:1.3e}")
print("\n")

# obtains λ' for each year
employ_coeffs = polynomial(years, employment_rate)
for i in range(len(employment_rate)):
    employ_deriv.append(derivative(employ_coeffs, i))

# obtains ω'/ω for each year
for i in range(len(gross_income)):
    y1.append(gross_deriv[i]/gross_income[i]) 
# obtains λ'/λ for each year
for i in range (len(employment_rate)):
    y2.append(employ_deriv[i]/employment_rate[i])

## First equation (ω'/ω = aλ + b)
# Coefs for the obtained equation 
a1, b1 = least_square_method(y1, employment_rate)
print("\n")
# Linear Regression
linear_regression(a1, b1, y1, employment_rate)
print("\n")

print("Employment Rate")
for i in range(len(employment_rate)):
    print(f"Value: {employment_rate[i]:1.3e} Derivative: {employ_deriv[i]:1.3e}")
print("\n")

## Second equation (λ'/λ = cω + d)
# Coefs for the obtained equation
a2, b2 = least_square_method(y2, gross_income)
print("\n")
# Linear Regression
linear_regression(a2, b2, y2, gross_income)
