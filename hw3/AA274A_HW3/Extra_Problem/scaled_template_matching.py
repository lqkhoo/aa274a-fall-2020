#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt


# Instead of returning tuples we return np.array
def match(template, image, scale, threshold):
    """
    """
<<<<<<< HEAD
    bboxes = set() # Just as an extra step make sure the boxes are unique.
=======
    bboxes = []
>>>>>>> d11a84634d5e3eb580d36f893a7c7b1ec41396fd
    I, F = image.astype(np.uint8), template.astype(np.uint8)
    h, w, _ = F.shape

    sim = cv2.matchTemplate(I, F, method=cv2.TM_CCORR_NORMED)
    x, y = np.nonzero(sim > threshold)
    n = x.shape[0] # number of boxes
    h = np.ones(n) * h
    w = np.ones(n) * w
    if n != 0:
        B = (np.vstack((x, y, h, w)).T  * scale).astype(np.int)
        for i in range(n):
<<<<<<< HEAD
            bboxes.add(tuple(B[i])) # Items need to be hashable.
=======
            bboxes.append(tuple(B[i]))
>>>>>>> d11a84634d5e3eb580d36f893a7c7b1ec41396fd
    return bboxes


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
    I_base = image

    # Attempt match at original scale
    # print(I_base.shape[0], I_base.shape[1])
    bboxes = match(F, I_base, scale=1.0, threshold=detection_threshold)
    matches.extend(bboxes)
    
    # Upscale
    I = I_base
    for k in range(num_upscales):
        I = cv2.pyrUp(I)
        bboxes = match(F, I, scale=1.0/(2**(k+1)), threshold=detection_threshold)
        matches.extend(bboxes)
    
    # Downscale
    I = I_base
    for k in range(num_downscales):
        I = cv2.pyrDown(I)
        bboxes = match(F, I, scale=(2**(k+1)), threshold=detection_threshold)
        matches.extend(bboxes)

    # Just as an extra step make sure the boxes are unique.
    matches = set(matches)

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

    matches = template_match(template, image, detection_threshold=0.93)

    """
    from pprint import pprint
    tmp = sorted(list(set(matches)))
    print(len(tmp))
    pprint(tmp)
    """
    create_and_save_detection_image(image, matches)

    template = cv2.imread('stop_signs/stop_template.jpg').astype(np.float32)
    for i in range(1, 6):
        image = cv2.imread('stop_signs/stop%d.jpg' % i).astype(np.float32)
        matches = template_match(template, image, detection_threshold=0.87)
        create_and_save_detection_image(image, matches, 'stop_signs/stop%d_detection.png' % i)

if __name__ == '__main__':
    main()
