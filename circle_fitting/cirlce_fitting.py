import cv2
import numpy as np
import math
import random

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = set([])


def fit_circle(data):
    """Use three data points and find the equation of a circle that bests fits the points"""

    if len(data) > 3:
        p = data[0]
        q = data[(len(data))/2]
        r = data[len(data)-1]
    elif len(data) == 3:
        p = data[0]
        q = data[1]
        r = data[2]

    matrix = np.array([[p[0], p[1], 1], [q[0], q[1], 1], [r[0], r[1], 1]])
    answer = np.array([-(p[0]**2 + p[1]**2), -(q[0]**2 + q[1]**2), -(r[0]**2 + r[1]**2)])

    unknowns = np.linalg.solve(matrix, answer)

    x_centre = float(unknowns[0])/2
    y_centre = float(unknowns[1])/2
    radius = math.sqrt(-unknowns[2] + x_centre**2 + y_centre**2)
    return -x_centre, -y_centre, radius


def ransac_circle(data):

    best_fit = []

    left_over = set(data)
    while len(left_over) >= 3:
        new_sample = set([])
        best = None
        besterr = float('-inf')
        for i in range(0, 200):
            sample = random.sample(left_over, 3)
            set_sample = set(sample)
            x_centre, y_centre, radius = fit_circle(sample)
            also_inliers = count_nearest_points(left_over - set(sample), x_centre, y_centre, radius)

            if len(also_inliers) >= 6:
                inliers = set_sample | also_inliers
                better_x, better_y, better_r = fit_circle(list(inliers))
                thiserr = count_nearest_points(inliers, better_x, better_y, better_r)

                if len(thiserr) > besterr:
                    best = [int(better_x), int(better_y), int(better_r)]
                    besterr = len(thiserr)
                new_sample = thiserr
        if len(new_sample) == 0:
            left_over = set([])
        else:
            left_over = left_over - new_sample
            best_fit.append(best)

    return best_fit


def count_nearest_points(data, x_centre, y_centre, radius):
    points_near = set([])
    for point in data:
        distance_from_center = math.sqrt((point[0] - x_centre) ** 2 + (point[1] - y_centre) ** 2)
        if distance_from_center <= radius**2:
            distance = abs(radius - distance_from_center)
        else:
            distance = abs(distance_from_center - radius)
        if distance <= 5:
            points_near.add(point)

    return points_near


def main():

    def click_point(event, x, y, flags, param):
        # grab references to the global variables
        global refPt, cropping

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt.add((x, y))
            print("Added point {}, {} to the data set.".format(x, y))
            # draw a rectangle around the region of interest
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("image", image)

    # x, y, r = fit_circle([(200, 55), (220, 56), (210, 66)])
    image = cv2.imread("../images/three_circles.png")
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_point)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(refPt) > 0:
        clone = image.copy()
        best_fit = ransac_circle(refPt)
        cv2.namedWindow("image")
        for fit in best_fit:
            cv2.circle(clone, (fit[0], fit[1]), fit[2], (0, 0, 255), 2)
        cv2.imshow("image", clone)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()