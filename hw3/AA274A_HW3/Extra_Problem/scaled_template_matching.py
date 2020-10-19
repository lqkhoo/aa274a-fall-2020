#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt


# Instead of returning tuples we return np.array
def match(template, image, threshold):
    """
    Input
        template: A (k, ell, c)-shaped ndarray containing the k x ell template (with c channels).
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        threshold: Minimum normalized cross-correlation value to be considered a match.

    Returns
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.
    """
    ########## Code starts here ##########
    bboxes = []
    I, F = image.astype(np.uint8), template.astype(np.uint8)
    match_method = cv2.TM_CCOEFF_NORMED
    f, g, _ = I.shape
    h, w, _ = F.shape

    sim = cv2.matchTemplate(I, F, match_method)
    x, y = np.nonzero(sim > threshold)
    n = x.shape[0] # number of boxes
    h = np.ones(n) * h
    w = np.ones(n) * w
    if n != 0:
        B = np.vstack((x, y, h, w)).astype(np.int).T
        for i in range(n):
            bboxes.append(B[i])

    return bboxes
    ########## Code ends here ##########


def template_match(template, image,
                   num_upscales=2, num_downscales=3,
                   detection_threshold=0.93):
    """
    Input
        template: A (k, ell, c)-shaped ndarray containing the k x ell template (with c channels).
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        num_upscales: How many times to 2x-upscale image with Gaussian blur before template matching over it.
        num_downscales: How many times to 0.5x-downscale image with Gaussian blur before template matching over it.
        detection_threshold: Minimum normalized cross-correlation value to be considered a match.

    Returns
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.
    """
    ########## Code starts here ##########
    matches = []

    F = template
    f, g, c = F.shape

    I_ori = image
    m_ori, n_ori, _ = I_ori.shape

    # Seek closest power of 2 greater than dimensions
    m = 1 << (m_ori-1).bit_length()
    n = 1 << (n_ori-1).bit_length()

    I = np.zeros((m, n, c))
    I[:m_ori, :n_ori] = I_ori # pad lower-right with zeros
    I_base = I
    # print(I_base.shape[0], I_base.shape[1])

    # Attempt match at original scale
    bboxes = match(F, I_base, threshold=detection_threshold)
    matches.extend(bboxes)
    
    # Upscale
    I = I_base
    for k in range(num_upscales):
        I = cv2.pyrUp(I)
        # m, n, _ = I.shape
        # I = cv2.pyrUp(I, dstsize=(n*2, m*2))
        # print(I.shape[0], I.shape[1])
        bboxes = match(F, I, threshold=detection_threshold)
        for i in range(len(bboxes)):
            bboxes[i] = bboxes[i] / (2**(k+1))
        matches.extend(bboxes)
    
    # Downscale
    I = I_base
    for k in range(num_downscales):
        I = cv2.pyrDown(I)
        # m, n, _ = I.shape
        # I = cv2.pyrDown(I, dstsize=(n/2, m/2))
        # print(I.shape[0], I.shape[1])
        bboxes = match(F, I, threshold=detection_threshold)
        for i in range(len(bboxes)):
            bboxes[i] = bboxes[i] * (2**(k+1))
        matches.extend(bboxes)

    return matches
    ########## Code ends here ##########


def create_and_save_detection_image(image, matches, filename="image_detections.png"):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.

    Returns
        None, this function will save the detection image in the current folder.
    """
    det_img = image.copy()
    for (y, x, bbox_h, bbox_w) in matches:
        cv2.rectangle(det_img, (x, y), (x + bbox_w, y + bbox_h), (255, 0, 0), 2)

    cv2.imwrite(filename, det_img)


def main():
    template = cv2.imread('messi_face.jpg')
    image = cv2.imread('messipyr.jpg')

    matches = template_match(template, image, detection_threshold=0.7)
    create_and_save_detection_image(image, matches)

    template = cv2.imread('stop_signs/stop_template.jpg').astype(np.float32)
    for i in range(1, 6):
        image = cv2.imread('stop_signs/stop%d.jpg' % i).astype(np.float32)
        matches = template_match(template, image, detection_threshold=0.87)
        create_and_save_detection_image(image, matches, 'stop_signs/stop%d_detection.png' % i)
    

if __name__ == '__main__':
    main()
