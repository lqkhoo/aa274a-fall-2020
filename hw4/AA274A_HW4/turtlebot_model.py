import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(x, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                        x: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to x.
        Gu: np.array[3,2] - Jacobian of g with respect ot u.
    """
    ########## Code starts here ##########

    # For derivations see answer to written HW4 Q1(i)

    X = x # Rename
    V, om = u       # u_t
    x, y, th = X    # x_{t-1}
    
    thomdt = th+om*dt # Common term

    if np.absolute(om) > EPSILON_OMEGA:
        # Common terms
        sin_minus_sin = np.sin(thomdt) - np.sin(th)
        cos_minus_cos = np.cos(thomdt) - np.cos(th)

        th_til  = th + om*dt
        x_til   = x + V/om * sin_minus_sin
        y_til   = y - V/om * cos_minus_cos

        dx_dth  = V/om * cos_minus_cos
        dy_dth  = V/om * sin_minus_sin
        dx_dV   = sin_minus_sin / om
        dy_dV   = -cos_minus_cos / om
        dx_dom  = V/(om*om) * (-sin_minus_sin + om*dt*np.cos(thomdt))
        dy_dom  = V/(om*om) * ( cos_minus_cos + om*dt*np.sin(thomdt))
        
    else:
        th_til  = th
        x_til   = x + V*dt*np.cos(th)
        y_til   = y + V*dt*np.sin(th)

        dx_dth  = -V*dt*np.sin(th)
        dy_dth  =  V*dt*np.cos(th)
        dx_dV   = dt*np.cos(th)
        dy_dV   = dt*np.sin(th)
        dx_dom  = -0.5*V*dt*dt*np.sin(th)
        dy_dom  =  0.5*V*dt*dt*np.cos(th)

    g = np.array([x_til, y_til, th_til]) # x_t = g is approximated by x_tilde
    Gx = np.array([
        [1, 0, dx_dth],
        [0, 1, dy_dth],
        [0, 0, 1]
    ], dtype=np.float)
    Gu = np.array([
        [dx_dV, dx_dom],
        [dy_dV, dy_dom],
        [0,     dt]
    ], dtype=np.float)

    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx


    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
