import cv2
import numpy as np
import math
import random

# initialize the list of reference points. This list will be populated by the points picked by the user
refPt = set([])


def ransac_circle(data):
    """RANSAC algorithm for fitting circles along a set of data points.
    :param data: Set of x, y points chosen py user, fitting a number of circles
    :returns best_fit: List of circle parameters deemed to create the best fits for the data"""

    best_fit = []                   # List of circle parameters deemed to create the best fits for the data

    left_over = set(data)
    while len(left_over) >= 3:      # left_over is the loop invariant that will decrease with each iteration
        new_sample = set([])        # The set of points that fit the best fit circle
        best = None
        best_count = float('-inf')    # set the best count to negative infinity, want to maximize this value
        for i in range(0, 200):
            # Pick three random points from data set
            sample = random.sample(left_over, 3)
            set_sample = set(sample)
            # Find best fit for the sample set
            x_centre, y_centre, radius = fit_circle(sample)
            # Check which points outside of the sample set fit into the circle model
            also_inliers = find_nearest_points(left_over - set(sample), x_centre, y_centre, radius)

            # Threshold a circle to only be valid if there are 6 more points that fit it
            if len(also_inliers) >= 6:
                inliers = set_sample | also_inliers
                this_count = find_nearest_points(inliers, x_centre, y_centre, radius)

                # Keep track of only the best
                if len(this_count) > best_count:
                    best = [int(x_centre), int(y_centre), int(radius)]
                    best_count = len(this_count)
                    new_sample = this_count
        if len(new_sample) == 0:
            left_over = set([])
        else:
            # Decrement the size of left over
            left_over = left_over - new_sample
            best_fit.append(best)

    return best_fit


def fit_circle(data):
    """Use three data points and find the equation of a circle that bests fits the points
    :param data: Set of x, y coordinates of length 3
    :returns x_centre: x value for the centre of the circle
    :returns y_centre: y value for the centre of the circle
    :returns radius: radius of the circle"""

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


def find_nearest_points(data, x_centre, y_centre, radius):
    """Find all the points that are within 5 pixels from the outline of the given circle.
    :param data: Set of x, y coordinates
    :param x_centre: The x coordinate of the centre of the circle
    :param y_centre: The y coordinate of the centre of the circle
    :param radius: The radius of the circle
    :returns Set of points in the data that are within 5 pixels of the circle"""

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
        global refPt

        # if the left mouse button was clicked, record the starting (x, y) coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt.add((x, y))
            print("Added point {}, {} to the data set.".format(x, y))
            # draw a circle around the region of interest
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("image", image)

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
