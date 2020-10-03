import numpy as np
import math
from numpy import linalg
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from utils import *

class State:
    def __init__(self,x,y,V,th):
        self.x = x
        self.y = y
        self.V = V
        self.th = th

    @property
    def xd(self):
        return self.V*np.cos(self.th)

    @property
    def yd(self):
        return self.V*np.sin(self.th)


def compute_traj_coeffs(initial_state, final_state, tf):
    """
    Inputs:
        initial_state (State)
        final_state (State)
        tf (float) final time
    Output:
        coeffs (np.array shape [8]), coefficients on the basis functions

    Hint: Use the np.linalg.solve function.
    """
    ########## Code starts here ##########

    x_0, x_tf       = initial_state.x,  final_state.x
    xdot_0, xdot_tf = initial_state.xd, final_state.xd
    y_0, y_tf       = initial_state.y,  final_state.y
    ydot_0, ydot_tf = initial_state.yd, final_state.yd

    # Since x,y are decoupled, we just solve them sequentially:
    # Ax = b_x,
    # Ay = b_y
    # To find: vectors x, y.

    A = np.array([
        [1, 0, 0, 0], # x(0)
        [0, 1, 0, 0], # xdot(0)
        [1, tf, tf*tf, tf*tf*tf], # x(tf)
        [0, 1, 2*tf,  3*tf*tf]   # xdot(tf)
    ])
    b_x = np.array([x_0, xdot_0, x_tf, xdot_tf])
    b_y = np.array([y_0, ydot_0, y_tf, ydot_tf])

    coeffs = np.zeros(8, dtype=float)
    coeffs[0:4] = np.linalg.solve(A, b_x)
    coeffs[4:] = np.linalg.solve(A, b_y)

    # assert(np.shape(coeffs) == (8,))

    ########## Code ends here ##########
    return coeffs

def compute_traj(coeffs, tf, N):
    """
    Inputs:
        coeffs (np.array shape [8]), coefficients on the basis functions
        tf (float) final_time
        N (int) number of points
    Output:
        traj (np.array shape [N,7]), N points along the trajectory, from t=0
            to t=tf, evenly spaced in time
    """
    t = np.linspace(0,tf,N) # generate evenly spaced points from 0 to tf
    traj = np.zeros((N,7))
    ########## Code starts here ##########

    x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4 = coeffs

    # For all required variables except theta, they are expressed as
    # a sum of basis functions, so let's first evaluate the basis functions
    # at the sampled points in time.

    # Theta is just arctan(ydot / xdot) due to the no-side-slip constraint.
    # We compute it last, since it depends on xdot and ydot.
    # The only challenge left is to vectorize the operation.

    psi = np.ones((4, N))
    psi[1] = t
    psi[2] = t * t
    psi[3] = t * t * t

    A = np.array([ # shape(7,4)
        [x_1,   x_2,   x_3,   x_4], # x(t)
        [y_1,   y_2,   y_3,   y_4], # y(t)
        [0,     0,     0,     0],   # \theta(t)
        [x_2,   2*x_3, 3*x_4, 0],   # xdot(t)
        [y_2,   2*y_3, 3*y_4, 0],   # ydot(t)
        [2*x_3, 6*x_4, 0,     0],   # xddot(t)
        [2*y_3, 6*y_4, 0,     0]    # yddot(t)
    ], dtype=float)

    traj = np.matmul(A, psi)

    ydot, xdot = traj[4], traj[3]
    traj[2] = np.arctan2(ydot, xdot)

    traj = traj.T # Question expects traj with shape (N,7)
    # assert(np.shape(traj) == (N,7))

    ########## Code ends here ##########

    return t, traj

def compute_controls(traj):
    """
    Input:
        traj (np.array shape [N,7])
    Outputs:
        V (np.array shape [N]) V at each point of traj
        om (np.array shape [N]) om at each point of traj
    """
    ########## Code starts here ##########

    N = traj.shape[0]

    # Split into individual trajectories of shape(1, N)
    traj = traj.T
    (x, y, theta, xdot, ydot, xddot, yddot) = traj

    # Cache
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # V could be calculated directly from xdot(t) = Vcos(\theta) etc etc.
    # We need to solve the matrix given by equation (2) to find \omega.

    ### Any of these should work.
    ### Actually first 2 options fail autograder for matrix singularity. Division by zero?
    # V = xdot / cos_theta
    # V = ydot / sin_theta
    V = np.sqrt(xdot*xdot + ydot*ydot)

    J = np.zeros((N, 2, 2))
    J[:,0,0] = cos_theta
    J[:,0,1] = -V * sin_theta
    J[:,1,0] = sin_theta
    J[:,1,1] = V * cos_theta

    b = np.vstack((xddot, yddot)).T

    # To solve: Jx = b
    soln = np.linalg.solve(J, b)
    a = soln[:,0]
    om = soln[:,1]

    # assert(np.shape(a) == (N,))
    # assert(np.shape(om) == (N,))

    ########## Code ends here ##########

    return V, om

def compute_arc_length(V, t):
    """
    This function computes arc-length s as a function of t.
    Inputs:
        V: a vector of velocities of length T
        t: a vector of time of length T
    Output:
        s: the arc-length as a function of time. s[i] is the arc-length at time
            t[i]. This has length T.

    Hint: Use the function cumtrapz. This should take one line.
    """
    s = None
    ########## Code starts here ##########
    s = cumtrapz(V, t, initial=0)
    ########## Code ends here ##########
    return s

def rescale_V(V, om, V_max, om_max):
    """
    This function computes V_tilde, given the unconstrained solution V, and om.
    Inputs:
        V: vector of velocities of length T. Solution from the unconstrained,
            differential flatness problem.
        om: vector of angular velocities of length T. Solution from the
            unconstrained, differential flatness problem.
    Output:
        V_tilde: Rescaled velocity that satisfies the control constraints.

    Hint: At each timestep V_tilde should be computed as a minimum of the
    original value V, and values required to ensure _both_ constraints are
    satisfied.
    Hint: This should only take one or two lines.
    """
    ########## Code starts here ##########

    ## V_tilde < 0.5 = V_max

    ## | om / V * V_tilde | < 1.0 = om_max
    ## V_tilde < | om_max / om * V |, trivial because V, V_tilde, om_max non-negative.

    V_tilde = np.minimum(V, V_max)
    V_tilde = np.minimum(V_tilde, np.abs(om_max / om * V))
    ########## Code ends here ##########
    return V_tilde


def compute_tau(V_tilde, s):
    """
    This function computes the new time history tau as a function of s.
    Inputs:
        V_tilde: a sequence of scaled velocities of length T.
        s: a sequence of arc-length of length T.
    Output:
        tau: the new time history for the sequence. tau[i] is the time at s[i]. This has length T.

    Hint: Use the function cumtrapz. This should take one line.
    """
    ########## Code starts here ##########
    tau = cumtrapz(1/V_tilde, s, initial=0)
    ########## Code ends here ##########
    return tau

def rescale_om(V, om, V_tilde):
    """
    This function computes the rescaled om control.
    Inputs:
        V: vector of velocities of length T. Solution from the unconstrained, differential flatness problem.
        om:  vector of angular velocities of length T. Solution from the unconstrained, differential flatness problem.
        V_tilde: vector of scaled velocities of length T.
    Output:
        om_tilde: vector of scaled angular velocities

    Hint: This should take one line.
    """
    ########## Code starts here ##########
    om_tilde = om / V * V_tilde
    ########## Code ends here ##########
    return om_tilde

def compute_traj_with_limits(z_0, z_f, tf, N, V_max, om_max):
    coeffs = compute_traj_coeffs(initial_state=z_0, final_state=z_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V,om = compute_controls(traj=traj)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)

    return traj, tau, V_tilde, om_tilde

def interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f):
    """
    Inputs:
        traj (np.array [N,7]) original unscaled trajectory
        tau (np.array [N]) rescaled time at orignal traj points
        V_tilde (np.array [N]) new velocities to use
        om_tilde (np.array [N]) new rotational velocities to use
        dt (float) timestep for interpolation

    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    """
    # Get new final time
    tf_new = tau[-1]

    # Generate new uniform time grid
    N_new = int(tf_new/dt)
    t_new = dt*np.array(range(N_new+1))

    # Interpolate for state trajectory
    traj_scaled = np.zeros((N_new+1,7))
    traj_scaled[:,0] = np.interp(t_new,tau,traj[:,0])   # x
    traj_scaled[:,1] = np.interp(t_new,tau,traj[:,1])   # y
    traj_scaled[:,2] = np.interp(t_new,tau,traj[:,2])   # th
    # Interpolate for scaled velocities
    V_scaled = np.interp(t_new, tau, V_tilde)           # V
    om_scaled = np.interp(t_new, tau, om_tilde)         # om
    # Compute xy velocities
    traj_scaled[:,3] = V_scaled*np.cos(traj_scaled[:,2])    # xd
    traj_scaled[:,4] = V_scaled*np.sin(traj_scaled[:,2])    # yd
    # Compute xy acclerations
    traj_scaled[:,5] = np.append(np.diff(traj_scaled[:,3])/dt,-s_f.V*om_scaled[-1]*np.sin(s_f.th)) # xdd
    traj_scaled[:,6] = np.append(np.diff(traj_scaled[:,4])/dt, s_f.V*om_scaled[-1]*np.cos(s_f.th)) # ydd

    return t_new, V_scaled, om_scaled, traj_scaled

if __name__ == "__main__":
    # traj, V, om = differential_flatness_trajectory()
    # Constants
    tf = 15.
    V_max = 0.5
    om_max = 1

    # time
    dt = 0.005
    N = int(tf/dt)+1
    t = dt*np.array(range(N))

    # Initial conditions
    s_0 = State(x=0, y=0, V=V_max, th=-np.pi/2)

    # Final conditions
    s_f = State(x=5, y=5, V=V_max, th=-np.pi/2)

    coeffs = compute_traj_coeffs(initial_state=s_0, final_state=s_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V,om = compute_controls(traj=traj)

    part_b_complete = False
    s = compute_arc_length(V, t)
    if s is not None:
        part_b_complete = True
        V_tilde = rescale_V(V, om, V_max, om_max)
        tau = compute_tau(V_tilde, s)
        om_tilde = rescale_om(V, om, V_tilde)

        print(traj.shape)
        print(tau.shape)
        print(V_tilde.shape)
        print(om_tilde.shape)
        print(dt)
        print(tf)

        t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)

        # Save trajectory data
        data = {'z': traj_scaled, 'V': V_scaled, 'om': om_scaled}
        save_dict(data, "data/differential_flatness.pkl")

    maybe_makedirs('plots')

    # Plots
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 2, 1)
    plt.plot(traj[:,0], traj[:,1], 'k-',linewidth=2)
    plt.grid(True)
    plt.plot(s_0.x, s_0.y, 'go', markerfacecolor='green', markersize=15)
    plt.plot(s_f.x, s_f.y, 'ro', markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title("Path (position)")
    plt.axis([-1, 6, -1, 6])

    ax = plt.subplot(2, 2, 2)
    plt.plot(t, V, linewidth=2)
    plt.plot(t, om, linewidth=2)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
    plt.title('Original Control Input')
    plt.tight_layout()

    plt.subplot(2, 2, 4, sharex=ax)
    if part_b_complete:
        plt.plot(t_new, V_scaled, linewidth=2)
        plt.plot(t_new, om_scaled, linewidth=2)
        plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
        plt.grid(True)
    else:
        plt.text(0.5,0.5,"[Problem iv not completed]", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.xlabel('Time [s]')
    plt.title('Scaled Control Input')
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    if part_b_complete:
        h, = plt.plot(t, s, 'b-', linewidth=2)
        handles = [h]
        labels = ["Original"]
        h, = plt.plot(tau, s, 'r-', linewidth=2)
        handles.append(h)
        labels.append("Scaled")
        plt.legend(handles, labels, loc="best")
    else:
        plt.text(0.5,0.5,"[Problem iv not completed]", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Arc-length [m]')
    plt.title('Original and scaled arc-length')
    plt.tight_layout()
    plt.savefig("plots/differential_flatness.png")
    plt.show()
