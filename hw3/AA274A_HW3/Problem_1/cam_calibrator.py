#!/usr/bin/env python

import pdb
import os
import sys

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from camera_calibration.calibrator import MonoCalibrator, ChessboardInfo, Patterns

class CameraCalibrator:

    def __init__(self):
        self.calib_flags = 0
        self.pattern = Patterns.Chessboard

    def loadImages(self, cal_img_path, name, n_corners, square_length, n_disp_img=1e5, display_flag=True):
        self.name = name
        self.cal_img_path = cal_img_path

        self.boards = []
        self.boards.append(ChessboardInfo(n_corners[0], n_corners[1], float(square_length)))
        self.c = MonoCalibrator(self.boards, self.calib_flags, self.pattern)

        if display_flag:
            fig = plt.figure('Corner Extraction', figsize=(12, 5))
            gs = gridspec.GridSpec(1, 2)
            gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            img = cv2.imread(self.cal_img_path + '/' + file, 0)     # Load the image
            img_msg = self.c.br.cv2_to_imgmsg(img, 'mono8')         # Convert to ROS Image msg
            drawable = self.c.handle_msg(img_msg)                   # Extract chessboard corners using ROS camera_calibration package

            if display_flag and i < n_disp_img:
                ax = plt.subplot(gs[0, 0])
                plt.imshow(img, cmap='gray')
                plt.axis('off')

                ax = plt.subplot(gs[0, 1])
                plt.imshow(drawable.scrib)
                plt.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                fig.canvas.set_window_title('Corner Extraction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

        # Useful parameters
        self.d_square = square_length                             # Length of a chessboard square
        self.h_pixels, self.w_pixels = img.shape                  # Image pixel dimensions
        self.n_chessboards = len(self.c.good_corners)             # Number of examined images
        self.n_corners_y, self.n_corners_x = n_corners            # Dimensions of extracted corner grid
        self.n_corners_per_chessboard = n_corners[0]*n_corners[1]

    def genCornerCoordinates(self, u_meas, v_meas):
        '''
        Inputs:
            u_meas: a list of arrays where each array are the u values for each board.
            v_meas: a list of arrays where each array are the v values for each board.
        Output:
            corner_coordinates: a tuple (Xg, Yg) where Xg/Yg is a list of arrays where
                                each array are the x/y values for each board.

        HINT: u_meas, v_meas starts at the blue end, and finishes with the pink end
        HINT: our solution does not use the u_meas and v_meas values
        HINT: it does not matter where your frame it, as long as you are consistent!
        '''
        ########## Code starts here ##########

        # u_meas, v_meas returns the corners in pixel coordinates, where
        # bottom-left is (0,0). Points are ordered from blue to pink.
        # Slanted lines show where the start of the next row is.

        # This is all good, but 
        # the Pset wants World coordinates (X,Y) attached to the
        # checkerboards, so we just generate these accordingly
        # based on what we know about the checkerboards.

        # In world coordinates (X,Y) in meters. The world reference frame
        # is centered on our arbitrary object. The way we're generating it,
        # X-axis goes from left-> right
        # Y-axis goes from top -> down
        # because that's how the points in u/v_meas are ordered in the array.

        N = self.n_chessboards
        nx, ny = self.n_corners_x, self.n_corners_y
        d = self.d_square
        # u_meas, v_meas have shapes (N, nx*ny)

        X_list = []
        Y_list = []

        # We don't assume nx, ny to be constant, so we need a loop
        for _ in range(N):
            Xs = np.multiply(d, np.arange(nx))
            Xs = np.tile(Xs, ny)    # Shape (nx*ny,) 1,2,3,4,1,2,3,4,1,2,3,4,...
            Ys = np.multiply(d, np.arange(ny))
            Ys = np.repeat(Ys, nx)  # Shape (nx*ny,) 1,1,1,2,2,2,3,3,3,4,4,4,...
            X_list.append(Xs)
            Y_list.append(Ys)

        corner_coordinates = (X_list, Y_list)

        ########## Code ends here ##########
        return corner_coordinates

    def estimateHomography(self, u_meas, v_meas, X, Y):    # Zhang Appendix A
        '''
        Inputs:
            u_meas: an array of the u values for a board.
            v_meas: an array of the v values for a board.
            X: an array of the X values for a board. (from genCornerCoordinates)
            Y: an array of the Y values for a board. (from genCornerCoordinates)
        Output:
            H: the homography matrix. its size is 3x3

        HINT: What is the size of the matrix L?
        HINT: What are the outputs of the np.linalg.svd function? Based on this,
                where does the eigenvector corresponding to the smallest eigen value live?
        HINT: np.stack and/or np.hstack may come in handy here.
        '''
        ########## Code starts here ##########

        # Zhang 1998. See also lecture 9 notes, especially section 9.2.1
        # Lecture 9 (2019) 54 minute mark onwards (Step 1)
        # Equation number (x) refer to lecture 9 notes.
        # We follow the notation in the lecture notes.

        # We need to reconstruct the homography matrix M mapping (homogeneous) world
        # coordinates to pixel coordinates.

        nx, ny = self.n_corners_x, self.n_corners_y
        u, v = u_meas, v_meas # shapes (nx*ny, )
        # X, Y as given, Z = 0 as we know the points are co-planar.

        P_W = np.array([X, Y, np.ones(nx*ny)]) # shape (k, nx*ny) = (3, 63)
        k = P_W.shape[0]

        # P_tilde has shape (2*nx*ny, 3k)
        # Each 'block' is [-1  0 ui ] * P_{W,i}^T
        #                 [ 0 -1 vi ]
        # and there are N of these blocks as we have N points per image.
        # P_{W,i} is a col vector of length 3.

        # This is not strictly matrix P_tilde as written in eq (18).
        # We blocked all rows with u and all the rows with v together.
        # Since each row is just a different constraint, this works just fine for SVD.
        P_tilde = np.block([
            [-1*P_W.T,  0*P_W.T, (u*P_W).T],
            [ 0*P_W.T, -1*P_W.T, (v*P_W).T]
        ])

        U, s, VT = np.linalg.svd(P_tilde, full_matrices=False) # We don't need padding.
        # Eigenvalues s sorted from largest to smallest.
        # We want the eigenvectors of P.T dot P, which corrresponds to V transpose.
        # Grab the smallest one in the last row, and reshape to M, which is 3x3.
        M = np.reshape(VT[-1, :], (k,k))

        H = M # Convert back to Zhang's terminology.

        ########## Code ends here ##########
        return H

    def compute_Vij(self, H_T, i, j):
        """
        Helper for getCameraIntrinsics.
        Expects H to already be transposed.
        arguments i and j should be cardinal (1-indexed)
        """
        i, j = i-1, j-1 # Translate to ordinal array index
        Hi, Hj = H_T[i], H_T[j]

        V_ij = np.array([
            Hi[0] * Hj[0],
            Hi[0] * Hj[1] + Hi[1] * Hj[0],
            Hi[1] * Hj[1],
            Hi[2] * Hj[0] + Hi[0] * Hj[2],
            Hi[2] * Hj[1] + Hi[1] * Hj[2],
            Hi[2] * Hj[2]
        ])

        return V_ij

    def getCameraIntrinsics(self, H):    # Zhang 3.1, Appendix B
        '''
        Input:
            H: a list of homography matrices for each board
        Output:
            A: the camera intrinsic matrix

        HINT: MAKE SURE YOU READ SECTION 3.1 THOROUGHLY!!! V. IMPORTANT
        HINT: What is the definition of h_ij?
        HINT: It might be cleaner to write an inner function (a function inside the getCameraIntrinsics function)
        HINT: What is the size of V?
        '''
        ########## Code starts here ##########

        # Equation numbers (x) refer to lecture 9 notes.

        H_list = H # Rebind variable name
        N = self.n_chessboards
        k = 3

        # Likewise, we will not replicate the matrix V exactly as in eq (27)
        # since it's just rows of SVD constraints.

        # First let's get rid of the list. (why use a list ?)
        # Each H is a 3x3, so let's turn the list into a tensor of shape (3, 3, N),
        # such that H[i, j] has shape (N, ).
        H_T = np.zeros((k, k, N))
        for n in range(N):
            H_T[:,:,n] = H_list[n].T # Note the transpose just under eq (26)

        V_11 = self.compute_Vij(H_T, 1, 1) # shape (2k, N) = (6, 23)
        V_12 = self.compute_Vij(H_T, 1, 2)
        V_22 = self.compute_Vij(H_T, 2, 2)

        V = np.vstack((
            V_12.T,
            (V_11-V_22).T
        )) # shape (2N, 2k) = (46, 6)

        U, s, VT_ = np.linalg.svd(V, full_matrices=False)
        b = VT_[-1, :]

        # We don't actually end up using the matrix B.
        # B = np.array([ # This is symmetric. We don't need the lower triangle.
        #     [b[0], b[1], b[3]],
        #     [b[1], b[2], b[4]],
        #     [b[3], b[4], b[5]]
        # ])

        # Then just plug in eq (30)

        (B11, B12, B22, B13, B23, B33) = b

        t1 = B12*B13 - B11*B23
        t2 = B11*B22 - B12*B12

        v0  = t1 / t2
        lmb = B33 - (B13*B13 + v0*t1) / B11
        alp = np.sqrt(lmb / B11)
        bet = np.sqrt(lmb * B11 / t2)
        gam = -B12 * alp*alp * bet / lmb
        u0  = gam * v0 / bet - B13 * alp*alp / lmb

        A = np.array([
            [alp, gam, u0],
            [0  , bet, v0],
            [0  , 0  , 1 ]
        ])

        ########## Code ends here ##########
        return A

    def getExtrinsics(self, H, A):    # Zhang 3.1, Appendix C
        '''
        Inputs:
            H: a single homography matrix
            A: the camera intrinsic matrix
        Outputs:
            R: the rotation matrix
            t: the translation vector
        '''
        ########## Code starts here ##########

        # Just follow equation (31) of lecture 9 notes
        A_inv = np.linalg.inv(A)
        c1, c2, c3 = H[:,0], H[:,1], H[:,2] # The cs are columns of H

        q = 1 / np.linalg.norm(np.dot(A_inv, c1)) # common recurring term
        r1 = q * np.dot(A_inv, c1)
        r2 = q * np.dot(A_inv, c2)
        r3 = np.cross(r1, r2)
        t  = q * np.dot(A_inv, c3)

        R = np.column_stack((r1, r2, r3)) # The rs are columns of R.

        U, s, VT = np.linalg.svd(R, full_matrices=False)
        R_sol = np.dot(U, VT)
        
        R = R_sol        

        ########## Code ends here ##########
        return R, t

    def transformWorld2NormImageUndist(self, X, Y, Z, R, t):    # Zhang 2.1, Eq. (1)
        '''
        Inputs:
            X, Y, Z: the world coordinates of the points for a given board. This is an array of 63 elements
                     X, Y come from genCornerCoordinates. Since the board is planar, we assume Z is an array of zeros.
            R, t: the camera extrinsic parameters (rotation matrix and translation vector) for a given board.
        Outputs:
            x, y: the coordinates in the ideal normalized image plane

        '''
        ########## Code starts here ##########
        
        d = X.shape[0]

        P_Wh = np.array([X, Y, Z, np.ones(d)])  # shape (4, 63) # Homogeneous world coords
        Rt = np.column_stack((R,t))             # shape (3, 4)
        P_Ch = np.dot(Rt, P_Wh)                 # shape (3, 63) # Homogeneous camera coords

        # Transform back to nonhomogeneous coords
        P_Ch = P_Ch / P_Ch[2]
        x, y, _ = P_Ch

        ########## Code ends here ##########
        return x, y

    def transformWorld2PixImageUndist(self, X, Y, Z, R, t, A):    # Zhang 2.1, Eq. (1)
        '''
        Inputs:
            X, Y, Z: the world coordinates of the points for a given board. This is an array of 63 elements
                     X, Y come from genCornerCoordinates. Since the board is planar, we assume Z is an array of zeros.
            A: the camera intrinsic parameters
            R, t: the camera extrinsic parameters (rotation matrix and translation vector) for a given board.
        Outputs:
            u, v: the coordinates in the ideal pixel image plane
        '''
        ########## Code starts here ##########

        d = X.shape[0]

        P_Wh = np.array([X, Y, Z, np.ones(d)])  # shape (4, 63) # Homogeneous world coords
        Rt = np.column_stack((R,t))             # shape (3, 4)
        P_Ch = np.matmul(Rt, P_Wh)              # shape (3, 63) # Homogeneous camera coords
        ph = np.matmul(A, P_Ch)                 # shape (3, 63) # Homogeneous pixel coords

        # Transform to nonhomogeneous coords
        ph = ph / ph[2]
        u, v, _ = ph
        
        ########## Code ends here ##########
        return u, v

    def undistortImages(self, A, k=np.zeros(2), n_disp_img=1e5, scale=0):
        Anew_no_k, roi = cv2.getOptimalNewCameraMatrix(A, np.zeros(4), (self.w_pixels, self.h_pixels), scale)
        mapx_no_k, mapy_no_k = cv2.initUndistortRectifyMap(A, np.zeros(4), None, Anew_no_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)
        Anew_w_k, roi = cv2.getOptimalNewCameraMatrix(A, np.hstack([k, 0, 0]), (self.w_pixels, self.h_pixels), scale)
        mapx_w_k, mapy_w_k = cv2.initUndistortRectifyMap(A, np.hstack([k, 0, 0]), None, Anew_w_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)

        if k[0] != 0:
            n_plots = 3
        else:
            n_plots = 2

        fig = plt.figure('Image Correction', figsize=(6*n_plots, 5))
        gs = gridspec.GridSpec(1, n_plots)
        gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            if i < n_disp_img:
                img_dist = cv2.imread(self.cal_img_path + '/' + file, 0)
                img_undist_no_k = cv2.undistort(img_dist, A, np.zeros(4), None, Anew_no_k)
                img_undist_w_k = cv2.undistort(img_dist, A, np.hstack([k, 0, 0]), None, Anew_w_k)

                ax = plt.subplot(gs[0, 0])
                ax.imshow(img_dist, cmap='gray')
                ax.axis('off')

                ax = plt.subplot(gs[0, 1])
                ax.imshow(img_undist_no_k, cmap='gray')
                ax.axis('off')

                if k[0] != 0:
                    ax = plt.subplot(gs[0, 2])
                    ax.imshow(img_undist_w_k, cmap='gray')
                    ax.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                fig.canvas.set_window_title('Image Correction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

    def plotBoardPixImages(self, u_meas, v_meas, X, Y, R, t, A, n_disp_img=1e5, k=np.zeros(2)):
        # Expects X, Y, R, t to be lists of arrays, just like u_meas, v_meas

        fig = plt.figure('Chessboard Projection to Pixel Image Frame', figsize=(8, 6))
        plt.clf()

        for p in range(min(self.n_chessboards, n_disp_img)):
            plt.clf()
            ax = plt.subplot(111)
            ax.plot(u_meas[p], v_meas[p], 'r+', label='Original')
            u, v = self.transformWorld2PixImageUndist(X[p], Y[p], np.zeros(X[p].size), R[p], t[p], A)
            ax.plot(u, v, 'b+', label='Linear Intrinsic Calibration')

            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height*0.85])
            ax.axis([0, self.w_pixels, 0, self.h_pixels])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title('Chessboard {0}'.format(p+1))
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize='medium', fancybox=True, shadow=True)

            plt.show(block=False)
            plt.waitforbuttonpress()

    def plotBoardLocations(self, X, Y, R, t, n_disp_img=1e5):
        # Expects X, U, R, t to be lists of arrays, just like u_meas, v_meas

        ind_corners = [0, self.n_corners_x-1, self.n_corners_x*self.n_corners_y-1, self.n_corners_x*(self.n_corners_y-1), ]
        s_cam = 0.02
        d_cam = 0.05
        xyz_cam = [[0, -s_cam, s_cam, s_cam, -s_cam],
                   [0, -s_cam, -s_cam, s_cam, s_cam],
                   [0, -d_cam, -d_cam, -d_cam, -d_cam]]
        ind_cam = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]
        verts_cam = []
        for i in range(len(ind_cam)):
            verts_cam.append([zip([xyz_cam[0][j] for j in ind_cam[i]],
                                  [xyz_cam[1][j] for j in ind_cam[i]],
                                  [xyz_cam[2][j] for j in ind_cam[i]])])

        fig = plt.figure('Estimated Chessboard Locations', figsize=(12, 5))
        axim = fig.add_subplot(121)
        ax3d = fig.add_subplot(122, projection='3d')

        boards = []
        verts = []
        for p in range(self.n_chessboards):

            M = []
            W = np.column_stack((R[p], t[p]))
            for i in range(4):
                M_tld = W.dot(np.array([X[p][ind_corners[i]], Y[p][ind_corners[i]], 0, 1]))
                if np.sign(M_tld[2]) == 1:
                    Rz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                    M_tld = Rz.dot(M_tld)
                    M_tld[2] *= -1
                M.append(M_tld[0:3])

            M = (np.array(M).T).tolist()
            verts.append([zip(M[0], M[1], M[2])])
            boards.append(Poly3DCollection(verts[p]))

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            if i < n_disp_img:
                img = cv2.imread(self.cal_img_path + '/' + file, 0)
                axim.imshow(img, cmap='gray')
                axim.axis('off')

                ax3d.clear()

                for j in range(len(ind_cam)):
                    cam = Poly3DCollection(verts_cam[j])
                    cam.set_alpha(0.2)
                    cam.set_color('green')
                    ax3d.add_collection3d(cam)

                for p in range(self.n_chessboards):
                    if p == i:
                        boards[p].set_alpha(1.0)
                        boards[p].set_color('blue')
                    else:
                        boards[p].set_alpha(0.1)
                        boards[p].set_color('red')

                    ax3d.add_collection3d(boards[p])
                    ax3d.text(verts[p][0][0][0], verts[p][0][0][1], verts[p][0][0][2], '{0}'.format(p+1))
                    plt.show(block=False)

                view_max = 0.2
                ax3d.set_xlim(-view_max, view_max)
                ax3d.set_ylim(-view_max, view_max)
                ax3d.set_zlim(-2*view_max, 0)
                ax3d.set_xlabel('X axis')
                ax3d.set_ylabel('Y axis')
                ax3d.set_zlabel('Z axis')

                if i == 0:
                    ax3d.view_init(azim=90, elev=120)

                plt.tight_layout()
                fig.canvas.set_window_title('Estimated Board Locations (Chessboard {0})'.format(i+1))

                plt.show(block=False)

                raw_input('<Hit Enter To Continue>')

    def undistortImages(self, A, k=np.zeros(2), n_disp_img=1e5, scale=0):
        Anew_no_k, roi = cv2.getOptimalNewCameraMatrix(A, np.zeros(4), (self.w_pixels, self.h_pixels), scale)
        mapx_no_k, mapy_no_k = cv2.initUndistortRectifyMap(A, np.zeros(4), None, Anew_no_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)
        Anew_w_k, roi = cv2.getOptimalNewCameraMatrix(A, np.hstack([k, 0, 0]), (self.w_pixels, self.h_pixels), scale)
        mapx_w_k, mapy_w_k = cv2.initUndistortRectifyMap(A, np.hstack([k, 0, 0]), None, Anew_w_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)

        if k[0] != 0:
            n_plots = 3
        else:
            n_plots = 2

        fig = plt.figure('Image Correction', figsize=(6*n_plots, 5))
        gs = gridspec.GridSpec(1, n_plots)
        gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            if i < n_disp_img:
                img_dist = cv2.imread(self.cal_img_path + '/' + file, 0)
                img_undist_no_k = cv2.undistort(img_dist, A, np.zeros(4), None, Anew_no_k)
                img_undist_w_k = cv2.undistort(img_dist, A, np.hstack([k, 0, 0]), None, Anew_w_k)

                ax = plt.subplot(gs[0, 0])
                ax.imshow(img_dist, cmap='gray')
                ax.axis('off')

                ax = plt.subplot(gs[0, 1])
                ax.imshow(img_undist_no_k, cmap='gray')
                ax.axis('off')

                if k[0] != 0:
                    ax = plt.subplot(gs[0, 2])
                    ax.imshow(img_undist_w_k, cmap='gray')
                    ax.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                fig.canvas.set_window_title('Image Correction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

    def writeCalibrationYaml(self, A, k):
        self.c.intrinsics = np.array(A)
        self.c.distortion = np.hstack(([k[0], k[1]], np.zeros(3))).reshape((1, 5))
        #self.c.distortion = np.zeros(5)
        self.c.name = self.name
        self.c.R = np.eye(3)
        self.c.P = np.column_stack((np.eye(3), np.zeros(3)))
        self.c.size = [self.w_pixels, self.h_pixels]

        filename = self.name + '_calibration.yaml'
        with open(filename, 'w') as f:
            f.write(self.c.yaml())

        print('Calibration exported successfully to ' + filename)

    def getMeasuredPixImageCoord(self):
        u_meas = []
        v_meas = []
        for chessboards in self.c.good_corners:
            u_meas.append(chessboards[0][:, 0][:, 0])
            # v_meas.append(chessboards[0][:, 0][:, 1]) # This keeps Y's original direction instead.
            v_meas.append(self.h_pixels - chessboards[0][:, 0][:, 1])   # Flip Y-axis to traditional direction

        return u_meas, v_meas   # Lists of arrays (one per chessboard)
