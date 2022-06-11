import seam_carving as sc
import cv2
import numpy as np
import image_quality_metrics as qm

COMPARE_PATH = 'images/compare/'
RESULTS_PATH = 'images/results/'
DIFF_PATH = 'images/differences/'

compare_1 = cv2.imread(COMPARE_PATH + '1_compare_crop_water.png')
compare_2 = cv2.imread(COMPARE_PATH + '2_compare_seam_enlarge_dolphin.png')
compare_3 = cv2.imread(COMPARE_PATH + '3_compare_enlarge_dolphin.png')
compare_4 = cv2.imread(COMPARE_PATH + '4_compare_enlarge_dolphin_2.png')
compare_5 = cv2.imread(COMPARE_PATH + '5_compare_seam_crop_bw_bench.png')
compare_6 = cv2.imread(COMPARE_PATH + '6_compare_crop_bw_bench.png')
compare_7 = cv2.imread(COMPARE_PATH + '7_compare_seam_crop_fw_bench.png')
compare_8 = cv2.imread(COMPARE_PATH + '8_compare_crop_fw_bench.png')
compare_9 = cv2.imread(COMPARE_PATH + '9_compare_extend_bw_car.png')
compare_10 = cv2.imread(COMPARE_PATH + '10_compare_extend_fw_car.png')

compare = [compare_1, compare_2, compare_3, compare_4, compare_5,
           compare_6, compare_7, compare_8, compare_9, compare_10]

results_1 = cv2.imread(RESULTS_PATH + '1_crop_water.png')
results_2 = cv2.imread(RESULTS_PATH + '2_seam_enlarge_dolphin.png')
results_3 = cv2.imread(RESULTS_PATH + '3_enlarge_dolphin.png')
results_4 = cv2.imread(RESULTS_PATH + '4_enlarge_dolphin_2.png')
results_5 = cv2.imread(RESULTS_PATH + '5_seam_crop_bw_bench.png')
results_6 = cv2.imread(RESULTS_PATH + '6_crop_bw_bench.png')
results_7 = cv2.imread(RESULTS_PATH + '7_seam_crop_fw_bench.png')
results_8 = cv2.imread(RESULTS_PATH + '8_crop_fw_bench.png')
results_9 = cv2.imread(RESULTS_PATH + '9_extend_bw_car.png')
results_10 = cv2.imread(RESULTS_PATH + '10_extend_fw_car.png')

results = [results_1, results_2, results_3, results_4, results_5,
           results_6, results_7, results_8, results_9, results_10]


for i in range(10):
    if qm.validate_img_shape(compare[i], results[i]):
        e_mse = qm.imMSE(compare[i], results[i])
        pSNR = qm.imPSNR(e_mse)
        print('i, e_mse, pSNR : ', i + 1, e_mse, pSNR)
        qm.histogram_comparison(compare[i], results[i])
        qm.visualize_diff(compare[i], results[i], DIFF_PATH + str(i + 1) + '_diff')
        print('\n')