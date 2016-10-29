import cv2
import numpy as np
import math

# initialize the list of reference points and boolean indicating
    # whether cropping is being performed or not
refPt = []


def fit_circle(data):
    """Use three data points and find the perpecicular bisector of two cords to find the equation of the circle"""

    if len(data) != 3:
        print("Incorrect data input size, should be of length 3, was length " + str(len(data)))
    else:
        p = data[0]
        q = data[1]
        r = data[2]

        matrix = np.array([[p[0], p[1], 1], [q[0], q[1], 1], [r[0], r[1], 1]])
        answer = np.array([-(p[0]**2 + p[1]**2), -(q[0]**2 + q[1]**2), -(r[0]**2 + r[1]**2)])

        unknowns = np.linalg.solve(matrix, answer)

        x_center = float(unknowns[0])/2
        y_center = float(unknowns[1])/2
        radius = math.sqrt(-unknowns[2] + x_center**2 + y_center**2)
        return -x_center, -y_center, radius


def main():

    def click_point(event, x, y, flags, param):
        # grab references to the global variables
        global refPt, cropping

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt.append((x, y))
            print("Added point {}, {} to the data set.".format(x, y))
            # draw a rectangle around the region of interest
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("image", image)
    fit_circle([(2, 1), (0, 5), (-1, 2)])

    image = cv2.imread("../images/three_circles.png")
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_point)

    cv2.imshow("image", image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()