import seam_carving as sc
import cv2
import numpy as np

# GET THE BASE IMAGES
img_bench = cv2.imread("images/base/bench.png")
img_car = cv2.imread("images/base/car.png")
img_dolphin = cv2.imread("images/base/dolphin.png")
img_water = cv2.imread("images/base/water.png")

# bw_energy = sc.backward_energy(img_dolphin)[0]
# sc.save_img('images/bw_energy', bw_energy)
#
# fw_energy = sc.forward_energy(img_dolphin)[0]
# sc.save_img('images/fw_energy', fw_energy)
#
# # GENERATE THE RESULTS
# # narrowed waterfall image with 50% seams removed
# crop_water, seam_crop_water, _ = sc.condense(img_water, 50, 'backward')
# sc.save_img('images/results/1_crop_water', crop_water)
# sc.save_img('images/results/1_seam_crop_water', seam_crop_water)
#
# # dolphin image with red seams added to enlarge by 50%
# enlarge_dolphin, seam_enlarge_dolphin = sc.extend(img_dolphin, 50, 'backward')
# sc.save_img('images/results/3_enlarge_dolphin', enlarge_dolphin)
# sc.save_img('images/results/2_seam_enlarge_dolphin', seam_enlarge_dolphin)
#
# # dolphin image with 2 steps of 50% insertion
# enlarge_dolphin_2, seam_enlarge_dolphin_2 = sc.extend(enlarge_dolphin.astype(np.uint8), 34, 'backward')
# sc.save_img('images/results/4_enlarge_dolphin_2', enlarge_dolphin_2)
# sc.save_img('images/results/4_seam_enlarge_dolphin_2', seam_enlarge_dolphin_2)
#
# bench with red seams to be removed by backward energy
crop_bw_bench, seam_crop_bw_bench, _ = sc.condense(img_bench, 50, 'backward')
sc.save_img('images/results/6_crop_bw_bench', crop_bw_bench)
sc.save_img('images/results/5_seam_crop_bw_bench', seam_crop_bw_bench)
#
# # bench with red seams removed by forward energy
# crop_fw_bench, seam_crop_fw_bench, _ = sc.condense(img_bench, 50, 'forward')
# sc.save_img('images/results/8_crop_fw_bench', crop_fw_bench)
# sc.save_img('images/results/7_seam_crop_fw_bench', seam_crop_fw_bench)
#
# # stretched car with seams inserted using backward energy
# extend_bw_car, seam_extend_bw_car = sc.extend(img_car, 50, 'backward')
# sc.save_img('images/results/9_extend_bw_car', extend_bw_car)
# sc.save_img('images/results/9_seam_extend_bw_car', seam_extend_bw_car)
#
# # stretched car with seams inserted using forward energy
# extend_fw_car, seam_extend_fw_car = sc.extend(img_car, 50, 'forward')
# sc.save_img('images/results/10_extend_fw_car', extend_fw_car)
# sc.save_img('images/results/10_seam_extend_fw_car', seam_extend_fw_car)
