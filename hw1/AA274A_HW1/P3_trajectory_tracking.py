import numpy as np
from numpy import linalg

V_PREV_THRES = 0.0001

class TrajectoryTracker:
    """ Trajectory tracking controller using differential flatness """
    def __init__(self, kpx, kpy, kdx, kdy, V_max=0.5, om_max=1):
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy

        self.V_max = V_max
        self.om_max = om_max

        self.coeffs = np.zeros(8) # Polynomial coefficients for x(t) and y(t) as
                                  # returned by the differential flatness code

    def reset(self):
        self.V_prev = 0
        self.om_prev = 0
        self.t_prev = 0

    def load_traj(self, times, traj):
        """ Loads in a new trajectory to follow, and resets the time """
        self.reset()
        self.traj_times = times
        self.traj = traj

    def get_desired_state(self, t):
        """
        Input:
            t: Current time
        Output:
            x_d, xd_d, xdd_d, y_d, yd_d, ydd_d: Desired state and derivatives
                at time t according to self.coeffs
        """
        x_d = np.interp(t,self.traj_times,self.traj[:,0])
        y_d = np.interp(t,self.traj_times,self.traj[:,1])
        xd_d = np.interp(t,self.traj_times,self.traj[:,3])
        yd_d = np.interp(t,self.traj_times,self.traj[:,4])
        xdd_d = np.interp(t,self.traj_times,self.traj[:,5])
        ydd_d = np.interp(t,self.traj_times,self.traj[:,6])
        
        return x_d, xd_d, xdd_d, y_d, yd_d, ydd_d

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time
        Outputs: 
            V, om: Control actions
        """

        dt = t - self.t_prev
        x_d, xd_d, xdd_d, y_d, yd_d, ydd_d = self.get_desired_state(t)

        ########## Code starts here ##########
        kpx, kdx, kpy, kdy = self.kpx, self.kdx, self.kpy, self.kdy

        V_d = np.sqrt(x_d*x_d + y_d*y_d) # V_nominal
        V = np.maximum(self.V_prev, V_PREV_THRES)
        ## print("V_d:{:5f} V:{:5f}".format(V_d, V))

        xdot = V * np.cos(th)
        ydot = V * np.sin(th)

        w_x = xdd_d + kpx*(x_d - x) + kdx*(xd_d - xdot)
        w_y = ydd_d + kpy*(y_d - y) + kdy*(yd_d - ydot)
        ## Characteristic equation A=1, B=k_1, C=k_2
        ## Critical damping when B^2=4AC = 0.

        w = np.array([w_x, w_y])
        J = np.array([
            [np.cos(th), -V*np.sin(th)],
            [np.sin(th), V*np.cos(th)]
        ])
        u = np.linalg.solve(J, w)

        a, om = u
        delta_v = a * (t - self.t_prev)
        V = self.V_prev + delta_v
        if V < V_PREV_THRES:
            v = np.maximum(V_d, V_PREV_THRES)
        ## print("t:{}, dv:{}".format(t, delta_v))

        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        return V, om