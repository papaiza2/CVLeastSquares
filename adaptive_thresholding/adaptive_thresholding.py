import sys
import cv2


def adaptive_threshold(img, type='adaptive', region=5):
    if type == 'mean':
        return _mean_thresholding(img, region)
    elif type == 'adaptive':
        return _adaptive_thresholding(img)
    else:
        print "Not a valid thresholding type"

def _mean_thresholding(img, region):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, region, 3)


def _adaptive_thresholding(img):
    pass


def main():
    # video_capture = cv2.VideoCapture(0)
    img = cv2.imread('../images/wedge.png')
    while True:
        # ret, frame = video_capture.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mean_image = adaptive_threshold(gray, type='mean', region=7)

        # Display the resulting frame
        cv2.imshow("Gray", gray)
        cv2.imshow('Mean Image', mean_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    # video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


