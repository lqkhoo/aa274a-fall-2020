#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt


def half_downscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        downscaled_image: A half-downscaled version of image.
    """
    ########## Code starts here ##########
    I = image
    I = I[::2, ::2, :] # Grab every other row and col by setting stride to 2
    return I
    ########## Code ends here ##########


def blur_half_downscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        downscaled_image: A half-downscaled version of image.
    """
    ########## Code starts here ##########
    I = image
    I = cv2.GaussianBlur(I, ksize=(5,5), sigmaX=0.7)
    I = I[::2, ::2, :]
    return I
    ########## Code ends here ##########


def two_upscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        upscaled_image: A 2x-upscaled version of image.
    """
    ########## Code starts here ##########
    I = image
    I = np.repeat(I, 2, axis=0)
    I = np.repeat(I, 2, axis=1)
    return I

    # Alternatively, probably less efficiently
    ## I = np.kron(image, np.ones((2,2)))
    ## return I

    ########## Code ends here ##########


def bilinterp_upscale(image, scale):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
        scale: How much larger to make the image

    Returns
        upscaled_image: A scale-times upscaled version of image.
    """
    m, n, c = image.shape

    f = (1./scale) * np.convolve(np.ones((scale, )), np.ones((scale, )))
    f = np.expand_dims(f, axis=0) # Making it (1, (2*scale)-1)-shaped
    filt = f.T * f

    ########## Code starts here ##########

    s = int(scale) # expects scale to be a power of 2.
    assert(s % 2 == 0)

    I, F = image, filt
    G = np.zeros((m*s-(s-1), n*s-(s-1), c)) # We don't want the last group of zeros
    G[::s, ::s, :] = I
    G = cv2.filter2D(G, -1, F)
    return G

    ########## Code ends here ##########


def main():
    # OpenCV actually uses a BGR color channel layout,
    # Matplotlib uses an RGB color channel layout, so we're flipping the 
    # channels here so that plotting matches what we expect for colors.
    test_card = cv2.imread('test_card.png')[..., ::-1].astype(float)
    favicon = cv2.imread('favicon-16x16.png')[..., ::-1].astype(float)
    test_card /= test_card.max()
    favicon /= favicon.max()

    # Note that if you call matplotlib's imshow function to visualize images,
    # be sure to pass in interpolation='none' so that the image you see
    # matches exactly what's in the data array you pass in.
    
    ########## Code starts here ##########

    _valid_cases = [
        'bilinear_upscale',
        'upscale',
        'blurred_halfscale',
        'halfscale'
    ]

    case = 'bilinear_upscale'

    img = None
    func = None
    file_name = None

    if case == 'bilinear_upscale':
        img = favicon
        func = bilinterp_upscale
        file_name = 'outputs/favicon_bilinear_8x.png'
        img = func(img, scale=2)

    elif case == 'upscale':
        img = favicon
        func = two_upscale
        file_name = 'outputs/favicon_upscale_8x.png'
        img = func(img)
        img = func(img)
        img = func(img)

    elif case == 'blur_halfscale':
        img = test_card
        func = blur_half_downscale
        file_name = 'outputs/test_card_blur_eighthscale.png'
        img = func(img)
        img = func(img)
        img = func(img)

    elif case == 'halfscale':
        img = test_card
        func = half_downscale
        file_name = 'outputs/test_card_eighthscale.png'
        img = func(img)
        img = func(img)
        img = func(img)

    else:
        raise ValueError("Case '{}' not understood.".format(case))

    fig, ax = plt.subplots()
    img = ax.imshow(img, interpolation='none')
    fig.savefig(file_name, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    ########## Code ends here ##########


if __name__ == '__main__':
    main()
