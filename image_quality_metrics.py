import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim


# arbitrary value
ABR = 1000000000

MAX_PIXEL_INTENSITY = 255


# MEAN-SQUARED ERROR (MSE)
# MSE measures the average squared difference between actual and ideal pixel values
def imMSE(compare, result):
    # arbitrarily value
    e_mse = ABR
    if compare.shape != result.shape:
        print('Error: The input images must have the same dimension')
    else:
        rows, cols = compare.shape[:2]

        # convert image to gray scale
        compare = cv2.cvtColor(compare, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.uint8)

        e_mse = (1 / float(rows * cols)) * (np.sum((compare - result) ** 2))

    return round(e_mse, 2)


# PEAK-SIGNAL-TO-NOISE-RATIO (pSNR)
# pSNR is derived from the mean square error and indicates the ratio of the maximum
# pixel intensity to the power of the distortion
def imPSNR(e_mse):
    pSNR = ABR
    if e_mse == 0:
        print('Warning: Images are the same (no SNR)')
    elif e_mse == ABR:
        print('Warning: e_mse is an arbitrary value ')
    else:
        # 255 is max pixel intensity
        pSNR = 10 * np.log(MAX_PIXEL_INTENSITY ** 2 / e_mse)  # pSNR is in dB

    return round(pSNR, 2)


# VALIDATE IMAGE SHAPE
def validate_img_shape(compare, result):
    print('c', compare.shape)
    print('r', result.shape)

    if compare.shape != result.shape:
        print('Error: The compare and result image do not have the same dimension')
        return False

    return True


# VISUALIZE DIFFERENT BETWEEN TWO IMAGES
def visualize_diff(compare, result, name):

    # ref: https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python

    compare = cv2.cvtColor(compare, cv2.COLOR_BGR2RGB)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Convert images to grayscale
    compare_gray = cv2.cvtColor(compare, cv2.COLOR_BGR2GRAY)
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    # score, diff = compare_ssim(compare_gray, result_gray, full=True)
    # print("Image similarity", score)
    diff = cv2.subtract(compare_gray, result_gray)
    diff = (diff * 255).astype(np.uint8)

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(compare, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.rectangle(result, (x, y), (x + w, y + h), (36, 255, 12), 2)

    titles = ['Compare', 'Result', 'Diff']
    images = [compare, result, diff]
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.savefig(name + '.png')
    plt.show()


# Histogram Comparison
def histogram_comparison(compare, result):
    # ref: https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html

    hsv_base = cv2.cvtColor(compare, cv2.COLOR_BGR2HSV)
    hsv_test1 = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    # Initialize the arguments to calculate the histograms (bins, ranges and channels H and S ).
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]

    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges  # concat lists

    # Use the 0-th and 1-st channels
    channels = [0, 1]

    # Calculate the Histograms for the compare image and the result image:
    hist_base = cv2.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_test1 = cv2.calcHist([hsv_test1], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Methods:
    # 0 - Correlation
    # 1 - Chi square
    # 2 - Intersection
    # 3 - Bhattacharyya

    for compare_method in range(4):
        base_base = cv2.compareHist(hist_base, hist_base, compare_method)
        base_test1 = cv2.compareHist(hist_base, hist_test1, compare_method)
        print('Method', compare_method, ' : ', 'Perfect ', '{:.2f}'.format(round(base_base, 2)), '\t',
              'Compare-Result', '{:.2f}'.format(round(base_test1, 2)))
