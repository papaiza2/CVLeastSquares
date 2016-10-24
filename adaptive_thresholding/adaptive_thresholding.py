import cv2
import numpy as np


def adaptive_threshold(img, type='adaptive', region=5):
    if type == 'mean':
        return _mean_thresholding(img)
    elif type == 'adaptive':
        return _least_squares_thresholding(img, region)
    else:
        print "Not a valid thresholding type"


def _mean_thresholding(img):
    sum = 0
    shape = img.shape
    for row in range(0, shape[0]):
        for col in range(0,shape[1]):
            sum += img[row, col]
    mean = sum/(shape[0]*shape[1])
    return threshold(img, mean)


def threshold(img, mean):
    new_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for row in range(0,img.shape[0]):
        for col in range(0,img.shape[1]):
            if img[row, col] > mean:
                new_img[row, col] = mean
            else:
                new_img[row,col] = img[row, col]
    return new_img


def _least_squares_thresholding(img, region):
    pass


def main():
    img = cv2.imread('../images/wedge.png')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean_image = adaptive_threshold(gray, type='mean')

    # Display the resulting frame
    cv2.imshow("Gray", gray)
    cv2.imshow('Mean Image', mean_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


