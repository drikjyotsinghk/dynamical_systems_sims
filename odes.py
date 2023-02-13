#Numerical Experiments with the Henon-Heiles Sytem of ODEs
#Dr. Ikjyot Singh Kohli - February, 2023
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#Define the Henon-Heiles System
def odes(t, y, l):
    dydt = np.zeros_like(y)
    dydt[0] = y[1]
    dydt[1] = -y[0] - 2*l*y[0]*y[2]
    dydt[2] = y[3]
    dydt[3] = y[2] - l*(y[0]**2 - y[2]**2)
    return dydt

# set initial conditions and time grid
y0 = [0,0.1,0.1,0.2]

t_span = [0,5]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

#parameters
l = 1

# solve the dynamical  system
sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, args = (l,), method="Radau")

# plot the solution - phase plot
plt.plot(sol.t, sol.y[0], label='x(t)')
plt.plot(sol.t, sol.y[2], label='y(t)')
plt.xlabel('t')
plt.legend()
plt.show()

#Experiment with Multiple Initial Conditions - difficult to maintain due to nonlinearity of system
#Need to manage initial conditions carefully
for i in range(11):
    y0 = np.random.rand(4)
    t_span = [0,2] # time interval to integrate over
    t_eval = np.linspace(t_span[0], t_span[1], 1000) # time grid for output

    #parameters
    l = 1

    # solve the ODEs
    sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, args = (1,), method="BDF")

    # plot the solution
    plt.plot(sol.t, sol.y[0], label='x(t)')
    plt.plot(sol.t, sol.y[2], label='y(t)')
    
plt.xlabel('x')
plt.ylabel('y')
plt.show()
