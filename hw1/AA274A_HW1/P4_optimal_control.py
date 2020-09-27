import numpy as np
import math
import scikits.bvp_solver
import matplotlib.pyplot as plt
from utils import *

dt = 0.005

def ode_fun(tau, z):
    """
    This function computes the dz given tau and z. It is used in the bvp solver.
    Inputs:
        tau: the independent variable. This must be the first argument.
        z: the state vector. The first three states are [x, y, th, ...]
    Output:
        dz: the state derivative vector. Returns a numpy array.
    """
    ########## Code starts here ##########
    (x, y, theta, p1, p2, p3, r) = z

    ### Terms in z are functions of time only. Not each other.
    ### H is a function of z

    ### From the Hamiltonian,
    # dH_dV = 2*V - p_1*V*np.sin(theta) + p_2*V*np.cos(theta)
    # dH_domega = 2*omega + p_3

    ### From dH_du = 0
    # dH_dV = 0
    # dH_domega = 0

    ### Therefore
    omega = - p_3 / 2.0

    dH_dx = 0 # H is not a function of x
    dH_dy = 0 # H is not a function of y
    dH_dtheta = -p1*V*np.sin(theta) + p2*V*np.cos(theta)

    x_dot = V * np.cos(theta)
    y_dot = V * np.sin(theta)
    theta_dot = omega
    p1_dot = -dH_dx
    p2_dot = -dH_dy
    p3_dot = -dH_dtheta
    r_dot = 0

    z_dot = np.array([x_dot, y_dot, theta_dot, p1_dot, p2_dot, p3_dot, r_dot])

    ### multiply by r to make dz derivative wrt new time tau = t/t_f
    dz = r * z_dot

    ########## Code ends here ##########
    return dz


def bc_fun(za, zb):
    """
    This function computes boundary conditions. It is used in the bvp solver.
    Inputs:
        za: the state vector at the initial time
        zb: the state vector at the final time
    Output:
        bca: tuple of boundary conditions at initial time
        bcb: tuple of boundary conditions at final time
    """
    # final goal pose
    xf = 5
    yf = 5
    thf = -np.pi/2.0
    xf = [xf, yf, thf]
    # initial pose
    x0 = [0, 0, -np.pi/2.0]

    ########## Code starts here ##########
    LAMBDA = 0.5

    x_0, y_0, theta_0, p1_0, p2_0, p3_0, r_0 = za
    x_f, y_f, theta_f, p1_f, p2_f, p3_f, r_f = zb

    bca = np.array([
        x_0 - x0[0],
        y_0 - x0[1],
        theta_0 - x0[2]
    ])

    omega = - p3_f / 2.0
    V = np.sqrt(x_f*x_f + y_f*y_f)

    bcb = np.array([
        x_f - xf[0],
        y_f - xf[1],
        theta_f - xf[2],
        LAMBDA + V*V + omega*omega + p1_f*V*np.sin(theta_f) +
                    p2_f*V*np.sin(theta_f) + p3_f*omega
    ])


    ########## Code ends here ##########
    return (bca, bcb)

def solve_bvp(problem_inputs, initial_guess):
    """
    This function solves the bvp_problem.
    Inputs:
        problem_inputs: a dictionary of the arguments needs to define the problem
                        num_ODE, num_parameters, num_left_boundary_conditions,
                        boundary_points, function, boundary_conditions
        initial_guess: initial guess of the solution
    Output:
        z: a numpy array of the solution. It is of size [time, state_dim]

    Read this documentation -- https://pythonhosted.org/scikits.bvp_solver/tutorial.html
    """
    problem = scikits.bvp_solver.ProblemDefinition(**problem_inputs)
    soln = scikits.bvp_solver.solve(problem, solution_guess=initial_guess)

    # Test if time is reversed in bvp_solver solution
    flip, tf = check_flip(soln(0))
    t = np.arange(0,tf,dt)
    z = soln(t/tf)
    if flip:
        z[3:7,:] = -z[3:7,:]
    z = z.T # solution arranged so that it is [time, state_dim]
    return z

def compute_controls(z):
    """
    This function computes the controls V, om, given the state z. It is used in main().
    Input:
        z: z is the state vector for multiple time instances. It has size [time, state_dim]
    Outputs:
        V: velocity control input
        om: angular rate control input
    """
    ########## Code starts here ##########



    ########## Code ends here ##########

    return V, om

def main():
    """
    This function solves the specified bvp problem and returns the corresponding optimal contol sequence
    Outputs:
        V: optimal V control sequence 
        om: optimal om ccontrol sequence
    You are required to define the problem inputs, initial guess, and compute the controls

    Hint: The total time is between 15-25
    """
    ########## Code starts here ##########

    num_ODE = 3+1
    num_parameters = 0
    num_left_boundary_conditions = 4
    boundary_points = (0, 1) # domain of integration \tau_0 = 0 and \tau = 1
    function = ode_fun
    boundary_conditions = bc_fun

    initial_guess = np.array([0, 0, 0, 0, 0], dtype=np.float)

    ########## Code ends here ##########

    problem_inputs = {
                      'num_ODE' : num_ODE,
                      'num_parameters' : num_parameters,
                      'num_left_boundary_conditions' : num_left_boundary_conditions,
                      'boundary_points' : boundary_points,
                      'function' : function,
                      'boundary_conditions' : boundary_conditions
                     }

    z = solve_bvp(problem_inputs, initial_guess)
    V, om = compute_controls(z)
    return z, V, om

if __name__ == '__main__':
    z, V, om = main()
    tf = z[0,-1]
    t = np.arange(0,tf,dt)
    x = z[:,0]
    y = z[:,1]
    th = z[:,2]
    data = {'z': z, 'V': V, 'om': om}
    save_dict(data, 'data/optimal_control.pkl')
    maybe_makedirs('plots')

    # plotting
    # plt.rc('font', weight='bold', size=16)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, y,'k-',linewidth=2)
    plt.quiver(x[1:-1:200], y[1:-1:200],np.cos(th[1:-1:200]),np.sin(th[1:-1:200]))
    plt.grid(True)
    plt.plot(0,0,'go',markerfacecolor='green',markersize=15)
    plt.plot(5,5,'ro',markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis([-1, 6, -1, 6])
    plt.title('Optimal Control Trajectory')

    plt.subplot(1, 2, 2)
    plt.plot(t, V,linewidth=2)
    plt.plot(t, om,linewidth=2)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc='best')
    plt.title('Optimal control sequence')
    plt.tight_layout()
    plt.savefig('plots/optimal_control.png')
    plt.show()
