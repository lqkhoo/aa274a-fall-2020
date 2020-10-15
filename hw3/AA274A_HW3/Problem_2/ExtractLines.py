#!/usr/bin/env python

############################################################
# ExtractLines.py
#
# This script reads in range data from a csv file, and
# implements a split-and-merge to extract meaningful lines
# in the environment.
############################################################

# Imports
import numpy as np
from PlotFunctions import *


############################################################
# functions
############################################################

def ExtractLines(RangeData, params):
    '''
    This function implements a split-and-merge line extraction algorithm.

    Inputs:
        RangeData: (x_r, y_r, theta, rho)
            x_r: robot's x position (m).
            y_r: robot's y position (m).
            theta: (1D) np array of angle 'theta' from data (rads).
            rho: (1D) np array of distance 'rho' from data (m).
        params: dictionary of parameters for line extraction.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        segend: np array (N_lines, 4) of line segment endpoints. Each row represents [x1, y1, x2, y2].
        pointIdx: (N_lines,2) segment's first and last point index.
    '''

    # Extract useful variables from RangeData
    x_r = RangeData[0]
    y_r = RangeData[1]
    theta = RangeData[2]
    rho = RangeData[3]

    ### Split Lines ###
    N_pts = len(rho)
    r = np.zeros(0)
    alpha = np.zeros(0)
    pointIdx = np.zeros((0, 2), dtype=np.int)

    # This implementation pre-prepartitions the data according to the "MAX_P2P_DIST"
    # parameter. It forces line segmentation at sufficiently large range jumps.
    rho_diff = np.abs(rho[1:] - rho[:(len(rho)-1)])
    LineBreak = np.hstack((np.where(rho_diff > params['MAX_P2P_DIST'])[0]+1, N_pts))
    startIdx = 0
    for endIdx in LineBreak:
        alpha_seg, r_seg, pointIdx_seg = SplitLinesRecursive(theta, rho, startIdx, endIdx, params)
        N_lines = r_seg.size

        ### Merge Lines ###
        if (N_lines > 1):
            alpha_seg, r_seg, pointIdx_seg = MergeColinearNeigbors(theta, rho, alpha_seg, r_seg, pointIdx_seg, params)
        r = np.append(r, r_seg)
        alpha = np.append(alpha, alpha_seg)
        pointIdx = np.vstack((pointIdx, pointIdx_seg))
        startIdx = endIdx

    N_lines = alpha.size

    ### Compute endpoints/lengths of the segments ###
    segend = np.zeros((N_lines, 4))
    seglen = np.zeros(N_lines)
    for i in range(N_lines):
        rho1 = r[i]/np.cos(theta[pointIdx[i, 0]]-alpha[i])
        rho2 = r[i]/np.cos(theta[pointIdx[i, 1]-1]-alpha[i])
        x1 = rho1*np.cos(theta[pointIdx[i, 0]])
        y1 = rho1*np.sin(theta[pointIdx[i, 0]])
        x2 = rho2*np.cos(theta[pointIdx[i, 1]-1])
        y2 = rho2*np.sin(theta[pointIdx[i, 1]-1])
        segend[i, :] = np.hstack((x1, y1, x2, y2))
        seglen[i] = np.linalg.norm(segend[i, 0:2] - segend[i, 2:4])

    ### Filter Lines ###
    # Find and remove line segments that are too short
    goodSegIdx = np.where((seglen >= params['MIN_SEG_LENGTH']) &
                          (pointIdx[:, 1] - pointIdx[:, 0] >= params['MIN_POINTS_PER_SEGMENT']))[0]
    pointIdx = pointIdx[goodSegIdx, :]
    alpha = alpha[goodSegIdx]
    r = r[goodSegIdx]
    segend = segend[goodSegIdx, :]

    # change back to scene coordinates
    segend[:, (0, 2)] = segend[:, (0, 2)] + x_r
    segend[:, (1, 3)] = segend[:, (1, 3)] + y_r

    return alpha, r, segend, pointIdx

def SplitLinesInner(theta, rho, startIdx, endIdx, alpha, r, idx, params):
    """
    Helper to SplitLinesRecursive
    """

    # Current subset
    a, b = startIdx, endIdx
    theta_sub = theta[a:b]
    rho_sub   = rho[a:b]

    # Fit a line parameterized by alpha and r to current subset.
    alpha_sub, r_sub = FitLine(theta_sub, rho_sub)
    if b - a < params['MIN_POINTS_PER_SEGMENT']: # Don't split anymore if we're already at minimum.
        alpha.append(alpha_sub)
        r.append(r_sub)
        idx.append((a,b))
        return

    # Otherwise determine the point to split.
    s = FindSplit(theta_sub, rho_sub, alpha_sub, r_sub, params)
    if s == -1: # No split
        alpha.append(alpha_sub)
        r.append(r_sub)
        idx.append((a,b))
        return
    else:
        SplitLinesInner(theta, rho, a, a+s, alpha, r, idx, params)
        SplitLinesInner(theta, rho, a+s, b, alpha, r, idx, params)
    

def SplitLinesRecursive(theta, rho, startIdx, endIdx, params):
    '''
    This function executes a recursive line-slitting algorithm, which
    recursively sub-divides line segments until no further splitting is
    required.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        startIdx: starting index of segment to be split.
        endIdx: ending index of segment to be split.
        params: dictionary of parameters.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        idx: (N_lines,2) segment's first and last point index.

    HINT: Call FitLine() to fit individual line segments.
    HINT: Call FindSplit() to find an index to split at.
    '''
    ########## Code starts here ##########

    alpha, r, idx = [], [], [] # lists because we're going to be appending things.
    SplitLinesInner(theta, rho, startIdx, endIdx, alpha, r, idx, params)
    alpha, r, idx = np.array(alpha), np.array(r), np.array(idx)

    ########## Code ends here ##########
    return alpha, r, idx

def FindSplit(theta, rho, alpha, r, params):
    '''
    This function takes in a line segment and outputs the best index at which to
    split the segment, or -1 if no split should be made.

    The best point to split at is the one whose distance from the line is
    the farthest, as long as this maximum distance exceeds
    LINE_POINT_DIST_THRESHOLD and also does not divide the line into segments
    smaller than MIN_POINTS_PER_SEGMENT. Otherwise, no split should be made.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: 'alpha' of input line segment (1 number).
        r: 'r' of input line segment (1 number).
        params: dictionary of parameters.
    Output:
        splitIdx: idx at which to split line (return -1 if it cannot be split).
    '''
    ########## Code starts here ##########

    D_THRESH = params['LINE_POINT_DIST_THRESHOLD']
    MIN_SEG  = params['MIN_POINTS_PER_SEGMENT']

    # If line is shorter than can be broken into two segments,
    # we'll never be able to find a split. We need this check, otherwise
    # we'll set the the whole of d to -1 and argmax will return 0, which is nonsense.
    n = theta.shape[0]
    if n < 2 * MIN_SEG:
        splitIdx = -1
    else:
        d = np.abs(rho * np.cos(theta - alpha) - r)
        # Ignore first MIN_SEG points at either end, because splitting there
        # violates the shortest segment criterion.
        d[-(MIN_SEG-1):] = -1   # Subtract 1 from end; e.g. if line is exactly 2*MIN_SEG,
                                # [:MIN_SEG], [MIN_SEG:] is a candidate split
        d[:MIN_SEG] = -1
        dmax_idx = np.argmax(d) # Tiebreaker is first element from the left.
        splitIdx = dmax_idx if d[dmax_idx] > D_THRESH else -1

    ########## Code ends here ##########
    return splitIdx

def FitLine(theta, rho):
    '''
    This function outputs a least squares best fit line to a segment of range
    data, expressed in polar form (alpha, r).

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
    Outputs:
        alpha: 'alpha' of best fit for range data (1 number) (rads).
        r: 'r' of best fit for range data (1 number) (m).
    '''
    ########## Code starts here ##########

    # We just follow eq (3) in pset.
    n = theta.shape[0] # num of points to fit

    # Handle the doubled sums first.
    # These variables only make sense within the doubled sum.
    theta_i, theta_j = np.repeat(theta, n), np.tile(theta, n)
    rho_i  , rho_j   = np.repeat(rho, n)  , np.tile(rho, n)

    t1 = np.sum(rho_i * rho_j * np.cos(theta_i) * np.sin(theta_j)) * 2.0 / n
    t2 = np.sum(rho_i * rho_j * np.cos(theta_i + theta_j)) / n

    num   = np.sum(rho * rho * np.sin(2 * theta)) - t1
    denom = np.sum(rho * rho * np.cos(2 * theta)) - t2

    alpha = 0.5 * np.arctan2(num, denom) + np.pi / 2.0
    r = np.sum(rho * np.cos(theta - alpha)) / n

    ########## Code ends here ##########
    return alpha, r

def MergeColinearNeigbors(theta, rho, alpha, r, pointIdx, params):
    '''
    This function merges neighboring segments that are colinear and outputs a
    new set of line segments.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        pointIdx: (N_lines,2) segment's first and last point indices.
        params: dictionary of parameters.
    Outputs:
        alphaOut: output 'alpha' of merged lines (rads).
        rOut: output 'r' of merged lines (m).
        pointIdxOut: output start and end indices of merged line segments.

    HINT: loop through line segments and try to fit a line to data points from
          two adjacent segments. If this line cannot be split, then accept the
          merge. If it can be split, do not merge.
    '''
    ########## Code starts here ##########

    # assert(alpha.shape[0] == r.shape[0] and r.shape[0] == pointIdx.shape[0])
    n = alpha.shape[0]
    
    # Since we have a few points only we'll just append to lists without preallocation.
    alphaOut, rOut, pointIdxOut = [], [], []

    for i in range(n):

        if i == 0:
            alphaOut.append(alpha[0])
            rOut.append(r[0])
            pointIdxOut.append(pointIdx[0])
            continue

        # Consider two segs together. We get the:
        a = min(pointIdxOut[-1][0], pointIdx[i][0]) # min of the start indices
        b = max(pointIdxOut[-1][1], pointIdx[i][1]) # max of the end indices
        theta_seg, rho_seg = theta[a:b], rho[a:b]
        alpha_seg, r_seg = FitLine(theta_seg, rho_seg)
        splitIdx = FindSplit(theta_seg, rho_seg, alpha_seg, r_seg, params)

        if splitIdx == -1:
            pass
        else:
            alphaOut.append(alpha[i])
            rOut.append(r[i])
            pointIdxOut.append(pointIdx[i])

    ########## Code ends here ##########
    return alphaOut, rOut, pointIdxOut


#----------------------------------
# ImportRangeData
def ImportRangeData(filename):

    data = np.genfromtxt('./RangeData/'+filename, delimiter=',')
    x_r = data[0, 0]
    y_r = data[0, 1]
    theta = data[1:, 0]
    rho = data[1:, 1]
    return (x_r, y_r, theta, rho)
#----------------------------------


############################################################
# Main
############################################################
def main():
    # parameters for line extraction (mess with these!)
    MIN_SEG_LENGTH = 0.05  # minimum length of each line segment (m)
    LINE_POINT_DIST_THRESHOLD = 0.02  # max distance of pt from line to split
    MIN_POINTS_PER_SEGMENT = 2  # minimum number of points per line segment
    MAX_P2P_DIST = 1.0  # max distance between two adjent pts within a segment

    # Data files are formated as 'rangeData_<x_r>_<y_r>_N_pts.csv
    # where x_r is the robot's x position
    #       y_r is the robot's y position
    #       N_pts is the number of beams (e.g. 180 -> beams are 2deg apart)

    # filename = 'rangeData_5_5_180.csv'
    # filename = 'rangeData_4_9_360.csv'
    filename = 'rangeData_7_2_90.csv'

    # Import Range Data
    RangeData = ImportRangeData(filename)

    params = {'MIN_SEG_LENGTH': MIN_SEG_LENGTH,
              'LINE_POINT_DIST_THRESHOLD': LINE_POINT_DIST_THRESHOLD,
              'MIN_POINTS_PER_SEGMENT': MIN_POINTS_PER_SEGMENT,
              'MAX_P2P_DIST': MAX_P2P_DIST}

    alpha, r, segend, pointIdx = ExtractLines(RangeData, params)

    ax = PlotScene()
    ax = PlotData(RangeData, ax)
    ax = PlotRays(RangeData, ax)
    ax = PlotLines(segend, ax)

    # plt.savefig('outputs/{}.{}'.format(filename, 'png'))
    plt.show(ax)

############################################################

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
