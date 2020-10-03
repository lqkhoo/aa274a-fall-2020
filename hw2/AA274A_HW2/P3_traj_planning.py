import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
import scipy.interpolate
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        ########## Code starts here ##########
        t_f = self.traj_controller.traj_times[-1]

        if t < t_f - self.t_before_switch:
            controller = self.traj_controller
        else:
            controller = self.pose_controller
        return controller.compute_control(x, y, th, t)
        ########## Code ends here ##########

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    tups = zip(*path)
    x = np.array(tups[0], dtype=np.float)
    y = np.array(tups[1], dtype=np.float)
    N = x.shape[0]

    dx = np.diff(x, n=1)
    dy = np.diff(y, n=1)

    l2 = np.sqrt(dx*dx + dy*dy) # l2 distance between each point.
    dt_ = l2 / V_des            # Time to travel between each segment at nominal speed.
    t = np.zeros(N)
    t[1:] = np.cumsum(dt_)      # Time at each point if travelling at nominal velocity
    t_f = t[-1]                 # Final time
    
    tck, u = scipy.interpolate.splprep([x, y], k=3, s=alpha) # Fit the spline

    subsample = True
    if subsample:
        N_SAMPLES = 512
        u = np.linspace(0, 1, N_SAMPLES)

    ts = u * t_f
    # Hence
    du_dt = 1 / t_f
    d2u_dt2 = 0

    xi, yi = scipy.interpolate.splev(u, tck, der=0)
    thetai = np.arctan2(yi, xi)
    xi_dot, yi_dot = scipy.interpolate.splev(u, tck, der=1)
    xi_dot = xi_dot * du_dt
    yi_dot = yi_dot * du_dt

    xi_ddot, yi_ddot = scipy.interpolate.splev(u, tck, der=2)
    xi_ddot = xi_ddot * d2u_dt2
    yi_ddot = yi_ddot * d2u_dt2
    
    traj_smoothed = np.array([xi, yi, thetai, xi_dot, yi_dot, xi_ddot, yi_ddot]).T
    
    t_smoothed = ts

    ########## Code ends here ##########

    return traj_smoothed, t_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    """
    ########## Code starts here ##########

    V, om = compute_controls(traj)

    # We need to grab the final state
    traj_t = traj.T
    x_f, y_f, theta_f = traj_t[0][-1], traj_t[1][-1], traj_t[2][-1]
    V_f = V[-1]
    s_f = State(x_f, y_f, V_f, theta_f)

    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)

    t_new, V_scaled, om_scaled, traj_scaled = (
                interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)
    )

    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled
