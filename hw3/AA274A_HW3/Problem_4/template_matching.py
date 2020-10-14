#!/usr/bin/env python

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt


def template_match(template, image, threshold=0.999):
    """
    Input
        template: A (k, ell, c)-shaped ndarray containing the k x ell template (with c channels).
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        threshold: Minimum normalized cross-correlation value to be considered a match.

    Returns
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.
    """
    ########## Code starts here ##########

    F, I, t = template, image, threshold
    f, g, c = F.shape
    m, n, _ = I.shape

    # Num of pixels to pad by
    px = int(np.floor(f / 2))
    py = int(np.floor(g / 2))

    I_padded = np.pad(I, ((px, px), (py, py), (0,0)), 'constant') # Don't pad channels
    bboxes = []

    # Naive loopy version
    cfg = c*f*g # length of our vectors
    Fvec = np.reshape(F, cfg, order='C')
    Fnorm = np.linalg.norm(Fvec)
    for u in range(m): # rows
        for v in range(n): # cols
            Ivec = I_padded[u:u+f, v:v+g, :] # (f, g, c)-shaped slice from padded image
            Ivec = np.reshape(Ivec, cfg, order='C')
            Inorm = np.linalg.norm(Ivec)
            # conditional to silence runtime warning if one of the norms is zero
            if np.all(Fnorm == 0) or np.all(Inorm == 0):
                sim = 0
            else:
                sim = np.dot(Fvec, Ivec) / Fnorm / Inorm
            if sim > t:
                bboxes.append((u-px, v-py, f, g))

    return bboxes
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

    cv2.imwrite(filename, 255*det_img/det_img.max())


def main():
    image = cv2.imread('clutter.png').astype(np.float32)
    template = cv2.imread('valdo.png').astype(np.float32)

    matches = template_match(template, image)
    create_and_save_detection_image(image, matches)

    template = cv2.imread('stop_signs/stop_template.jpg').astype(np.float32)
    for i in range(1, 6):
        image = cv2.imread('stop_signs/stop%d.jpg' % i).astype(np.float32)
        matches = template_match(template, image, threshold=0.85)
        create_and_save_detection_image(image, matches, 'stop_signs/stop%d_detection.png' % i)


if __name__ == "__main__":
    main()
