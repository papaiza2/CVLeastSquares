import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []


def click_point(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        # draw a rectangle around the region of interest
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("image", image)

image = cv2.imread("../images/three_circles.png")
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_point)

cv2.imshow("image", image)
cv2.waitKey(0)

cv2.destroyAllWindows()

