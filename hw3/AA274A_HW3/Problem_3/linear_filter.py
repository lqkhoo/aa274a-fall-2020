#!/usr/bin/env python

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt


def corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the correlation of the filter with the image.
    """
    ########## Code starts here ##########  

    # rename k, ell to f, g
    f, g, c = F.shape
    m, n, _ = I.shape

    # Num of pixels to pad by
    px = int(np.floor(f / 2))
    py = int(np.floor(g / 2))

    # For our convenience, for filters of even dimension we bias to
    # the lower or right cell as center. This means we just 'convolve'
    # the padded image m across and n down, using the top-left coordinate
    # as reference for the filter according to the Pset,
    # starting from (0,0) at top left of the image.

    I_padded = np.pad(I, ((px, px), (py, py), (0,0)), 'constant') # Don't pad channels
    
    # Naive loopy version
    # runtime .79, 1.36, .77, .80
    cfg = c*f*g # length of our vectors
    Fvec = np.reshape(F, cfg, order='C')
    G = np.zeros((m, n))
    for u in range(m): # rows
        for v in range(n): # cols
            Ivec = I_padded[u:u+f, v:v+g, :] # (f, g, c)-shaped slice from padded image
            Ivec = np.reshape(Ivec, cfg, order='C')
            G[u,v] = np.dot(Fvec, Ivec)
    return G

    """
    # Implementation with array indexing
    # Runtimes 0.13, 7.55, 0.12, 0.34
    # The required mem footprint will probably fail the autograder
    # since we are pre-calculating all the indices.
    
    Fvec = np.reshape(F, c*f*g, order='C')
    
    # Generate indices
    x_starts = np.arange(0, m)   # All possible start positions
    x_ends   = np.arange(f, f+m) # All possible end positions
    x_idxs = np.linspace(x_starts, x_ends, num=f, axis=1, endpoint=False, dtype=np.int)
    x_idxs = np.repeat(x_idxs, g, axis=1) # This accounts for height of filter
    x_idxs = np.repeat(x_idxs, n, axis=0) # Repeat y times
    # Shape should be (m*n, f*g). For each point in mxn image, we need to select f*g coords.

    # The last two repeats are equivalent to taking the Kronecker product, but repeats take less compute time.
    # x_idxs = np.kron(x_idxs, np.ones((n,g), dtype=np.int)) 

    y_starts = np.arange(0, n)   # All possible start positions
    y_ends   = np.arange(g, g+n) # All possible end positions
    y_idxs = np.linspace(y_starts, y_ends, num=g, axis=1, endpoint=False, dtype=np.int) # This goes [0,1,2], [1,2,3], [2,3,4],...
    y_idxs = np.tile(y_idxs, (m, f)) # Tile x times and account for width of filter
    # Shape should be (m*n, f*g).

    Ivec = I_padded[x_idxs, y_idxs, :] # shape (m*n, f, g)
    Ivec = np.reshape(Ivec, (m*n, -1), order="C") # Shape (m*n, f*g)
    G = np.dot(Ivec, Fvec)
    G = np.reshape(G, (m, n))
    return G
    """

    ########## Code ends here ##########


def norm_cross_corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the normalized cross-correlation of the filter with the image.
    """
    ########## Code starts here ##########

    # rename k, ell to f, g
    f, g, c = F.shape
    m, n, _ = I.shape

    # Num of pixels to pad by
    px = int(np.floor(f / 2))
    py = int(np.floor(g / 2))

    # For our convenience, for filters of even dimension we bias to
    # the lower or right cell as center. This means we just 'convolve'
    # the padded image m across and n down, using the top-left coordinate
    # as reference for the filter according to the Pset,
    # starting from (0,0) at top left of the image.

    I_padded = np.pad(I, ((px, px), (py, py), (0,0)), 'constant') # Don't pad channels
    
    # Naive loopy version
    cfg = c*f*g # length of our vectors
    Fvec = np.reshape(F, cfg, order='C')
    Fnorm = np.linalg.norm(Fvec)
    G = np.zeros((m, n))
    for u in range(m): # rows
        for v in range(n): # cols
            Ivec = I_padded[u:u+f, v:v+g, :] # (f, g, c)-shaped slice from padded image
            Ivec = np.reshape(Ivec, cfg, order='C')
            Inorm = np.linalg.norm(Ivec)
            G[u,v] = np.dot(Fvec, Ivec) / Fnorm / Inorm
    return G

    ########## Code ends here ##########


def show_save_corr_img(filename, image, template):
    # Not super simple, because need to normalize image scale properly.
    fig, ax = plt.subplots()
    cropped_img = image[:-template.shape[0], :-template.shape[1]]
    im = ax.imshow(image, interpolation='none', vmin=cropped_img.min())
    fig.colorbar(im)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def main():
    test_card = cv2.imread('test_card.png').astype(np.float32)

    filt1 = np.zeros((3, 3, 1))
    filt1[1, 1] = 1

    filt2 = np.zeros((3, 200, 1))
    filt2[1, -1] = 1

    filt3 = np.zeros((3, 3, 1))
    filt3[:, 0] = -1
    filt3[:, 2] = 1

    filt4 = (1./273.)*np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]])
    filt4 = np.expand_dims(filt4, -1)

    grayscale_filters = [filt1, filt2, filt3, filt4]

    color_filters = list()
    for filt in grayscale_filters:
        # Making color filters by replicating the existing
        # filter per color channel.
        color_filters.append(np.concatenate([filt, filt, filt], axis=-1))

    for idx, filt in enumerate(color_filters):
        start = time.time()
        corr_img = corr(filt, test_card)
        stop = time.time()
        print 'Correlation function runtime:', stop - start, 's'
        show_save_corr_img("corr_img_filt%d.png" % idx, corr_img, filt)


if __name__ == "__main__":
    main()
