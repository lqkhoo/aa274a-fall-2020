import numpy as np
import matplotlib.pyplot as plt
from utils import plot_line_segments, line_line_intersection

class RRT(object):
    """ Represents a motion planning problem to be solved using the RRT algorithm"""
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles):
        self.statespace_lo = np.array(statespace_lo)    # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)    # state space upper bound (e.g., [5, 5])
        self.x_init = np.array(x_init)                  # initial state
        self.x_goal = np.array(x_goal)                  # goal state
        self.obstacles = obstacles                      # obstacle set (line segments)
        self.path = None        # the final path as a list of states

    def is_free_motion(self, obstacles, x1, x2):
        """
        Subject to the robot dynamics, returns whether a point robot moving
        along the shortest path from x1 to x2 would collide with any obstacles
        (implemented as a "black box")

        Inputs:
            obstacles: list/np.array of line segments ("walls")
            x1: start state of motion
            x2: end state of motion
        Output:
            Boolean True/False
        """
        raise NotImplementedError("is_free_motion must be overriden by a subclass of RRT")

    def find_nearest(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the steering distance (subject to robot dynamics) from
        V[i] to x is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V to x
        """
        raise NotImplementedError("find_nearest must be overriden by a subclass of RRT")

    def steer_towards(self, x1, x2, eps):
        """
        Steers from x1 towards x2 along the shortest path (subject to robot
        dynamics). Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards must be overriden by a subclass of RRT")


    def sample_random_state(self):
        """
        Added helper method. Returns random state vector within valid bounds.
        For this exercise, we are just using uniform random sampling.
        """
        lo, hi, D = self.statespace_lo, self.statespace_hi, len(self.x_init)
        return np.squeeze(np.random.uniform(lo, hi, size=(1, D)))

    def trace_path(self, V, P, cur_state_idx):
        """
        Added helper method. Constructs path from given state to root node
        and sets self.path as equal to that path.
        """
        if self.path is None:
            self.path = []
        
        idx = cur_state_idx
        while idx != -1:
            self.path.append(V[idx])
            idx = P[idx]
        
        self.path = list(reversed(self.path))


    def solve(self, eps, max_iters=1000, goal_bias=0.05, shortcut=False):
        """
        Constructs an RRT rooted at self.x_init with the aim of producing a
        dynamically-feasible and obstacle-free trajectory from self.x_init
        to self.x_goal.

        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT iterations (early termination
                is possible when a feasible solution is found)
            goal_bias: probability during each iteration of setting
                x_rand = self.x_goal (instead of uniformly randly sampling
                from the state space)
        Output:
            None officially (just plots), but see the "Intermediate Outputs"
            descriptions below
        """

        state_dim = len(self.x_init)

        # V stores the states that have been added to the RRT (pre-allocated at its maximum size
        # since numpy doesn't play that well with appending/extending)
        V = np.zeros((max_iters, state_dim))
        V[0,:] = self.x_init    # RRT is rooted at self.x_init
        n = 1                   # the current size of the RRT (states accessible as V[range(n),:])

        # P stores the parent of each state in the RRT. P[0] = -1 since the root has no parent,
        # P[1] = 0 since the parent of the first additional state added to the RRT must have been
        # extended from the root, in general 0 <= P[i] < i for all i < n
        P = -np.ones(max_iters, dtype=int)

        success = False

        ## Intermediate Outputs
        # You must update and/or populate:
        #    - V, P, n: the represention of the planning tree
        #    - success: whether or not you've found a solution within max_iters RRT iterations
        #    - self.path: if success is True, then must contain list of states (tree nodes)
        #          [x_init, ..., x_goal] such that the global trajectory made by linking steering
        #          trajectories connecting the states in order is obstacle-free.

        ## Hints:
        #   - use the helper functions find_nearest, steer_towards, and is_free_motion
        #   - remember that V and P always contain max_iters elements, but only the first n
        #     are meaningful! keep this in mind when using the helper functions!

        ########## Code starts here ##########
        self.path = []
        for k in range(max_iters):

            p = np.random.uniform(0, 1)

            # An epsilon-random sampled point.
            x_rand = self.x_goal if p < goal_bias else self.sample_random_state()
            # Point in the tree that is closest (by distance metric) to x_rand.
            x_near_idx = self.find_nearest(V[range(n),:], x_rand)
            x_near = V[x_near_idx]
            # Candidate point to be added to the tree if we detect no collisions.
            x_new = self.steer_towards(x_near, x_rand, eps)

            if self.is_free_motion(self.obstacles, x_near, x_new):
                V[n,:] = x_new              # Add vertex
                x_new_idx = n
                P[x_new_idx] = x_near_idx   # Add edge between x_new to x_near.
                n += 1                      # Increment tree size

                if np.all(x_new == self.x_goal): # Breaking condition
                    self.trace_path(V, P, n-1)   # n points to empty space at the moment.
                    success = True
                    break

                

        ########## Code ends here ##########

        plt.figure()
        self.plot_problem()
        self.plot_tree(V, P, color="blue", linewidth=.5, label="RRT tree", alpha=0.5)
        if success:
            if shortcut:
                self.plot_path(color="purple", linewidth=2, label="Original solution path")
                self.shortcut_path()
                self.plot_path(color="green", linewidth=2, label="Shortcut solution path")
            else:
                self.plot_path(color="green", linewidth=2, label="Solution path")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
            plt.scatter(V[:n,0], V[:n,1])
        else:
            print "Solution not found!"

        return success

    def plot_problem(self):
        plot_line_segments(self.obstacles, color="red", linewidth=2, label="obstacles")
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
        plt.axis('scaled')

    def shortcut_path(self):
        """
        Iteratively removes nodes from solution path to find a shorter path
        which is still collision-free.
        Input:
            None
        Output:
            None, but should modify self.path
        """
        ########## Code starts here ##########
        
        path = self.path
        k = len(path)
        
        # This is our doubly-linked list.
        # When we remove a node,
        # we reassign the node's current child to the node's parent,
        # and the node's current parent to the node's child.
        # idx is zero-indexed sequence of nodes in non-shortcut path.
        P = np.linspace(0, k-1, k, dtype=np.int) - 1 # P[n] returns the idx of parent of nth node in path
        C = np.linspace(0, k-1, k, dtype=np.int) + 1 # C[n] returns the idx of child of nth node in path

        for i in range(k):
            # print("Current i: {}".format(i))
            # print("P {}".format(P))
            # print("C {}".format(C))

            node = path[i]
            if np.all(node == self.x_init) or np.all(node == self.x_goal):
                continue

            parent = path[P[i]]
            child = path[C[i]]
            if self.is_free_motion(self.obstacles, parent, child):
                # Drop current node. This means:
                P[C[i]] = P[i] # The parent of our child is now our parent (it was us)
                C[P[i]] = C[i] # The child of our parent is now our child (it was us)
                P[i] = -2      # This is just for notational clarity. Removed nodes don't have 
                C[i] = -2      # either parent or child.
                continue

        # Once we're done, just trace the children till we reach the end.
        shortcut = []
        idx = 0
        node = path[idx]
        shortcut.append(node)
        while True:
            idx = C[idx]
            if idx == len(C):
                break
            else:
                node = path[idx]
                shortcut.append(node)

        self.path = shortcut
        
        ########## Code ends here ##########

class GeometricRRT(RRT):
    """
    Represents a geometric planning problem, where the steering solution
    between two points is a straight line (Euclidean metric)
    """

    def find_nearest(self, V, x):
        ########## Code starts here ##########
        # Hint: This should take one line.

        # This function expects V to have been already trimmed down to size
        # without the zero entries.
        # Returns the *index* of the closest node, not the node itself.
        return np.argmin(np.linalg.norm(V - x, ord=2, axis=1))
        ########## Code ends here ##########

    def steer_towards(self, x1, x2, eps):
        ########## Code starts here ##########
        # Hint: This should take one line.
        dx = np.linalg.norm(x1 - x2, ord=2)
        if dx < eps:
            x = x2
        else:
            x = x1 + eps*(x2 - x1) / dx # Linear interpolation
        return x
        ########## Code ends here ##########

    def is_free_motion(self, obstacles, x1, x2):
        motion = np.array([x1, x2])
        for line in obstacles:
            if line_line_intersection(motion, line):
                return False
        return True

    def plot_tree(self, V, P, **kwargs):
        plot_line_segments([(V[P[i],:], V[i,:]) for i in range(V.shape[0]) if P[i] >= 0], **kwargs)

    def plot_path(self, **kwargs):
        path = np.array(self.path)
        plt.plot(path[:,0], path[:,1], **kwargs)

class DubinsRRT(RRT):
    """
    Represents a planning problem for the Dubins car, a model of a simple
    car that moves at a constant speed forward and has a limited turning
    radius. We will use this v0.9.2 of the package at
    https://github.com/AndrewWalker/pydubins/blob/0.9.2/dubins/dubins.pyx
    to compute steering distances and steering trajectories. In particular,
    note the functions dubins.path_length and dubins.path_sample (read
    their documentation at the link above). See
    http://planning.cs.uiuc.edu/node821.html
    for more details on how these steering trajectories are derived.
    """
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles, turning_radius):
        self.turning_radius = turning_radius
        super(self.__class__, self).__init__(statespace_lo, statespace_hi, x_init, x_goal, obstacles)

    def find_nearest(self, V, x):
        from dubins import path_length
        ########## Code starts here ##########
        min_path_length = np.inf
        min_path_idx = None
        for i in range(V.shape[0]):
            # For kinodynamic models, the distance function is generally not symmetric.
            # This has to be from V to x.
            le = path_length(np.ndarray.squeeze(V[i]), x, self.turning_radius)
            if le < min_path_length:
                min_path_length = le
                min_path_idx = i
        return min_path_idx
        ########## Code ends here ##########

    def steer_towards(self, x1, x2, eps):
        ########## Code starts here ##########
        """
        A subtle issue: if you use dubins.path_sample to return the point
        at distance eps along the path from x to y, use a turning radius
        slightly larger than self.turning_radius
        (i.e., 1.001*self.turning_radius). Without this hack,
        dubins.path_sample might return a point that can't quite get to in
        distance eps (using self.turning_radius) due to numerical precision
        issues.
        """
        from dubins import path_sample, path_length
        rho = 1.001 * self.turning_radius
        dx = path_length(x1, x2, rho)
        if dx < eps:
            x = x2
        else:
            samples, _ = path_sample(x1, x2, rho, eps)
            x = samples[1]
        return x
        ########## Code ends here ##########

    def is_free_motion(self, obstacles, x1, x2, resolution = np.pi/6):
        from dubins import path_sample
        pts = path_sample(x1, x2, self.turning_radius, self.turning_radius*resolution)[0]
        pts.append(x2)
        for i in range(len(pts) - 1):
            for line in obstacles:
                if line_line_intersection([pts[i][:2], pts[i+1][:2]], line):
                    return False
        return True

    def plot_tree(self, V, P, resolution = np.pi/24, **kwargs):
        from dubins import path_sample
        line_segments = []
        for i in range(V.shape[0]):
            if P[i] >= 0:
                pts = path_sample(V[P[i],:], V[i,:], self.turning_radius, self.turning_radius*resolution)[0]
                pts.append(V[i,:])
                for j in range(len(pts) - 1):
                    line_segments.append((pts[j], pts[j+1]))
        plot_line_segments(line_segments, **kwargs)

    def plot_path(self, resolution = np.pi/24, **kwargs):
        from dubins import path_sample
        pts = []
        path = np.array(self.path)
        for i in range(path.shape[0] - 1):
            pts.extend(path_sample(path[i], path[i+1], self.turning_radius, self.turning_radius*resolution)[0])
        plt.plot([x for x, y, th in pts], [y for x, y, th in pts], **kwargs)
