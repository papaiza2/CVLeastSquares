import cv2
import numpy as np


def adaptive_threshold(img, type='adaptive'):
    if type == 'mean':
        return _mean_thresholding(img)
    elif type == 'adaptive':
        return _least_squares_thresholding(img)
    else:
        print "Not a valid thresholding type"


def _mean_thresholding(img):
    sum = 0
    height, width = img.shape[0], img.shape[1]
    for row in range(0, height):
        for col in range(0, width):
            sum += img[row, col]
    mean = sum/(width*height)
    return threshold(img, mean)


def threshold(img, mean):
    height, width = img.shape[0], img.shape[1]
    new_img = np.zeros((height, width, 3), np.uint8)
    for row in range(0, height):
        for col in range(0, width):
            if img[row, col] > mean:
                new_img[row, col] = mean
            else:
                new_img[row, col] = img[row, col]
    return new_img


def _least_squares_thresholding(img):
    """Want to find the parameters a, b, c such that it minimizes the error for the linear system:
        a + bi + cj - I[i,j] = 0
        To solve for the unknowns create a matrix:

        | a_11 a_12 a_13 | |a|   |b_1|
        | a_21 a_22 a_23 | |b| = |b_2|
        | a_31 a_32 a_33 | |c|   |b_3|

        """

    height, width = img.shape[0], img.shape[1]
    a_11 = height * width
    a_12 = a_13 = a_21 = a_22 = a_23 = a_31 = a_32 = a_33 = 0
    b_1 = b_2 = b_3 = 0
    # i = row # , j = column #
    for i in range(0, height):
        a_12 += i + 1
        a_22 += (i+1)**2
    a_21 = a_12

    for j in range(0, width):
        a_13 += j + 1
        a_33 += (j+1)**2
    a_31 = a_13

    for i in range(0, height):
        for j in range(0, width):
            a_23 += (i+1)*(j+1)
            b_1 += img[i, j]
            b_2 += (i + 1) * img[i, j]
            b_3 += (j + 1) * img[i, j]
    a_32 = a_23

    matrix = np.array([[a_11, a_12, a_13], [a_21, a_22, a_23], [a_31, a_32, a_33]])
    answer = np.array([b_1, b_2, b_3])

    unknowns  = np.linalg.solve(matrix, answer)
    new_img = np.zeros((height, width, 3), np.uint8)

    for i in range(0, height):
        for j in range(0, width):
            new_img[i, j] = unknowns[0] + unknowns[1]*(i+1) + unknowns[2]*(j+1)

    return new_img


def main():
    img = cv2.imread('../images/tables.png')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean_image = adaptive_threshold(gray, type='adaptive')

    # Display the resulting frame
    cv2.imshow("Gray", gray)
    cv2.imshow('Mean Image', mean_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


