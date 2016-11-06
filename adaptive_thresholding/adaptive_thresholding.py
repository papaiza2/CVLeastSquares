import cv2
import numpy as np
import itertools


def adaptive_threshold(img, type='adaptive'):
    if type == 'mean':
        return _mean_thresholding(img)
    elif type == 'adaptive':
        return _least_squares_thresholding(img)
    else:
        print("Not a valid thresholding type")


def _mean_thresholding(img):
    mean = cv2.mean(img)[0]
    ret, new_img = cv2.threshold(img, mean, 255, cv2.THRESH_BINARY)
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
    a_11 = float(height * width)
    a_12 = a_13 = a_21 = a_22 = a_23 = a_31 = a_32 = a_33 = 0.0
    b_1 = b_2 = b_3 = 0.0

    a_21 = a_12 = float(height*(height + 1)/2) * width
    a_22 = float(height * (height + 1) * (2*height + 1)/6) * width

    a_31 = a_13 = float(width*(width + 1)/2) * height
    a_33 = float(width * (width + 1) * (2*width + 1)/6) * height
    a_32 = a_23 = float((height*(height + 1)/2) * (width*(width + 1)/2))

    b_1 = float(cv2.sumElems(img)[0])
    # b_2 = float(height*(height + 1)/2 * b_1)
    # b_3 = float(width*(width + 1)/2 * b_1)

    for j in range(0, width):
        for i in range(0, height):
            b_2 += float((i + 1) * img[i, j])
            b_3 += float((j + 1) * img[i, j])


    matrix = np.array([[a_11, a_12, a_13], [a_21, a_22, a_23], [a_31, a_32, a_33]])
    answer = np.array([b_1, b_2, b_3])

    unknowns = np.linalg.solve(matrix, answer)

    fitted_img = np.zeros((height, width), np.uint8)
    final_img = np.zeros((height, width), np.uint8)

    first = np.dot((unknowns[1] * np.arange(1, height + 1)).reshape(height, 1), np.ones((1, width)))
    second = np.dot(np.ones((height, 1)), (unknowns[2] * np.arange(1, width+1)).reshape(1, width))
    fitted_img = abs(unknowns[0] + np.add(first, second))

    for i, j in itertools.product(range(height), range(width)):
        # fitted_img[i, j] = unknowns[0] + unknowns[1] * (i + 1) + unknowns[2] * (j + 1)
        if img[i, j] + 15 > fitted_img[i, j]:
            final_img[i, j] = 255
        else:
            final_img[i, j] = 0
    # final_img = cv2.absdiff(img, fitted_img.astype(np.uint8))

    return fitted_img, final_img


def main():
    # cap = cv2.VideoCapture(0)
    # while(True):
    img = cv2.imread('../images/shadowed_page.png')
    # ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh, new_image = adaptive_threshold(gray, type='adaptive')

    # Display the resulting frame
    # cv2.imshow("Gray", gray)
    # cv2.imshow('Thresh', thresh)
    cv2.imshow('New', new_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


