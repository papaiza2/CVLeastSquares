import cv2
import numpy as np


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

    i_matrix = np.dot(np.arange(1, height+1).reshape(height, 1), np.ones((1, width)))
    j_matrix = np.dot(np.ones((height, 1)), np.arange(1, width + 1).reshape(1, width))
    b_2 = cv2.sumElems(np.multiply(i_matrix, img))[0]
    b_3 = cv2.sumElems(np.multiply(j_matrix, img))[0]

    matrix = np.array([[a_11, a_12, a_13], [a_21, a_22, a_23], [a_31, a_32, a_33]])
    answer = np.array([b_1, b_2, b_3])

    unknowns = np.linalg.solve(matrix, answer)

    first = np.dot((unknowns[1] * np.arange(1, height + 1)).reshape(height, 1), np.ones((1, width)))
    second = np.dot(np.ones((height, 1)), (unknowns[2] * np.arange(1, width+1)).reshape(1, width))
    fitted_img = unknowns[0] + np.add(first, second)

    final_img = cv2.absdiff(img + 235, fitted_img.astype(np.uint8))
    final_img = 255 - final_img
    ret, final_img = cv2.threshold(final_img, 215, 255, cv2.THRESH_BINARY)
    return fitted_img, final_img


def main():

    # For image

    # img = cv2.imread('../images/shadowed_page.png')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh, new_image = adaptive_threshold(gray, type='adaptive')
    # mean = adaptive_threshold(gray, type='mean')
    # cv2.imshow('Mean', mean)
    # cv2.imshow('Adaptive', new_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # For camera

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(15, 0.1)
    while (True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh, new_image = adaptive_threshold(gray, type='adaptive')
        cv2.imshow('New', new_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


