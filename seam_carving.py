import numpy as np
import cv2
from numba import jit


def save_img(name, img):
    cv2.imwrite(name + '.png', img.astype(np.uint8))


@jit
# using built-in sobel edge detection algorithm to find the backward energy map
def backward_energy(img):
    rows, cols = img.shape[:2]
    # Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    img = cv2.GaussianBlur(img, (3, 3), 0, borderType=cv2.BORDER_CONSTANT)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_CONSTANT)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_CONSTANT)

    # energy(I) = |dI/dx| + |dI/dy|

    # this give a better dolphin and bench result
    energy_map = np.abs(grad_x) + np.abs(grad_y)

    # this give a better car result
    # abs_grad_x = cv2.convertScaleAbs(grad_x)
    # abs_grad_y = cv2.convertScaleAbs(grad_y)
    # energy_map = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # M is the cumulative minimal energy map
    M = np.zeros_like(energy_map, dtype=np.float64)
    # deep copy the minimum energy of the last row to M
    M = energy_map.copy()
    # backtrack stores the column index of min energy
    backtrack = np.zeros((rows, cols), dtype=np.int)

    # calculate the backward energy accumulative matrix M
    # M[i, j] = e[i, j] + min(M[i - 1, j - 1], M[i - 1, j], M[i - 1, j + 1])
    for i in range(1, rows):
        for j in range(0, cols):
            left, right = max(0, j - 1), min(j + 2, cols)
            if j == 0:
                col_index = j + np.argmin(M[i - 1, left:right])
            else:
                col_index = j + np.argmin(M[i - 1, left:right]) - 1
            backtrack[i, j] = col_index
            M[i, j] += M[i - 1, col_index]

    return energy_map, M, backtrack


@jit
def forward_energy(img):
    rows, cols = img.shape[:2]
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    energy = np.zeros((rows, cols))
    M = np.zeros((rows, cols))
    # backtrack stores the column index of min energy
    backtrack = np.zeros((rows, cols), dtype=np.int)

    for i in range(0, rows):
        for j in range(cols):

            up = i - 1
            if j == 0:
                left, right = 0, j + 1
            elif j + 1 == cols:
                left, right = j - 1, j
            else:
                left, right = j - 1, j + 1

            # the cost respective to Left, Up, and Right neighbors
            cost_up = np.abs(I[i, right] - I[i, left])
            cost_left = cost_up + np.abs(I[up, j] - I[i, left])
            cost_right = cost_up + np.abs(I[up, j] - I[i, right])

            cost = np.array([cost_left, cost_up, cost_right])

            # forward energy accumulative cost matrix M
            total = np.array([M[up, left] + cost_left,
                              M[up, j] + cost_up,
                              M[up, right] + cost_right])
            if j == 0:
                col_index = j + np.argmin(total)
            else:
                col_index = j + np.argmin(total) - 1

            backtrack[i, j] = col_index
            M[i, j] = min(total)

            # energy at each pixel is the min cost relating to that pixel
            energy[i, j] = cost[np.argmin(total)]

    return energy, M, backtrack


@jit
def find_optimal_vertical_seam(M, backtrack):
    rows, cols = M.shape[:2]

    # find the optimal seam path, seam with the lowest energy
    seam = np.zeros((rows, 1), dtype=np.uint32)
    mask = np.ones_like(M, dtype=np.bool)

    # get the index of the smallest element in the last row of M
    col = np.argmin(M[-1])
    for row in reversed(range(rows)):
        seam[row] = col
        mask[row, col] = False
        col = backtrack[row, col]

    return seam, mask


def remove_seam(img, seam_mask):
    rows, cols = img.shape[:2]

    # convert mask to 3D array
    seam_mask = np.stack([seam_mask] * 3, axis=2)

    # remove seam using mask
    new_img = img[seam_mask].reshape((rows, cols - 1, 3))

    return new_img


@jit
def insert_seam(img, seam):
    rows, cols, chans = img.shape

    # increase the image by 1 col
    new_img = np.zeros((rows, cols + 1, chans))

    for chan in range(chans):
        for row, col in enumerate(seam):
            col = int(col)
            # the duplicated pixel is the average of their left and right neighbors
            if col == 0:
                new_img[row, 0, chan] = np.average(img[row, :2, chan])
                new_img[row, 1:, chan] = img[row, col:, chan]
            else:
                new_img[row, :col, chan] = img[row, :col, chan]
                new_img[row, col, chan] = np.average(img[row, col - 1:col + 1, chan])
                new_img[row, col + 1:, chan] = img[row, col:, chan]

    return new_img


@jit
def draw_seam(img, seam):
    rows, cols, chans = img.shape

    # increase the image by 1 col
    seam_img = np.zeros((rows, cols + 1, chans))

    # add red seam to image, algo is similar to insert seam
    for chan in range(chans):
        for row, col in enumerate(seam):
            col = int(col)

            # add red seam to image (BGR)
            if chan == 2:
                content = 255
            else:
                content = 0

            if col == 0:
                seam_img[row, 0, chan] = content
                seam_img[row, 1:, chan] = img[row, col:, chan]
            else:
                seam_img[row, :col, chan] = img[row, :col, chan]
                seam_img[row, col, chan] = content
                seam_img[row, col + 1:, chan] = img[row, col:, chan]

    return seam_img.astype(np.uint8)


@jit
def condense(img, factor, energy_function='backward'):
    rows, cols = img.shape[:2]

    # find the number of seams to remove base on input factor
    n_seams = int(np.floor(cols * factor / 100))

    new_img = img.copy()

    # save the list of seams to remove for use in extend() image function
    list_of_seams = []
    for i in range(n_seams):

        # calculate the energy map, min cumulative minimal energy map (M),
        # and the backtrack from minimal entry on M
        if energy_function == 'backward':
            energy_map, M, backtrack = backward_energy(new_img)
        else:
            energy_map, M, backtrack = forward_energy(new_img)

        # use backtrack to find the path of optimal seam
        seam, mask = find_optimal_vertical_seam(M, backtrack)

        list_of_seams.append(seam)

        # remove seam
        new_img = remove_seam(new_img, mask)

    # show the removed seams on the original image in red
    # add red seams from reduced size to original size to show all removed seams
    seam_img = new_img.copy()
    for seam in reversed(list_of_seams):
        seam_img = draw_seam(seam_img, seam)

    return new_img, seam_img, list_of_seams


@jit
def extend(img, factor, energy_function='backward'):
    seam_img = img.copy()
    new_img = img.copy()

    # get the list of optimal seams from the condense() function
    list_of_seams = condense(img, factor, energy_function)[2]

    # insert seams starting with the most (first found) optimal seam
    for i in range(len(list_of_seams)):
        current_seam = list_of_seams.pop(0)
        # insert seam
        new_img = insert_seam(new_img, current_seam)
        # draw seam
        seam_img = draw_seam(seam_img, current_seam)
        # update seam indices (still need more work...)
        list_of_seams = update_seam(list_of_seams, current_seam)

    return new_img, seam_img


@jit
def update_seam(queue, current):
    update = []
    for seam in queue:
        # use this to update each pixel (cause disjoint in seam)
        # seam[np.where(seam >= current)] += 1

        # shift the whole seam relative to the image size
        # (work slightly better but still incorrect)
        if seam[0] > current[0]:
            seam += 2

        update.append(seam)

    return update


if __name__ == "__main__":
    # GET THE BASE IMAGES
    img_bench = cv2.imread("images/base/bench.png")
    img_car = cv2.imread("images/base/car.png")
    img_dolphin = cv2.imread("images/base/dolphin.png")
    img_water = cv2.imread("images/base/water.png")

    bw_energy = backward_energy(img_dolphin)[0]
    save_img('images/bw_energy', bw_energy)

    fw_energy = forward_energy(img_dolphin)[0]
    save_img('images/fw_energy', fw_energy)

    # narrowed waterfall image with 50% seams removed
    crop_water, seam_crop_water, _ = condense(img_water, 50, 'backward')
    save_img('images/results/1_crop_water', crop_water)
    save_img('images/results/1_seam_crop_water', seam_crop_water)

    # dolphin image with red seams added to enlarge by 50%
    enlarge_dolphin, seam_enlarge_dolphin = extend(img_dolphin, 50, 'backward')
    save_img('images/results/3_enlarge_dolphin', enlarge_dolphin)
    save_img('images/results/2_seam_enlarge_dolphin', seam_enlarge_dolphin)

    # dolphin image with 2 steps of 50% insertion
    enlarge_dolphin_2, seam_enlarge_dolphin_2 = extend(enlarge_dolphin.astype(np.uint8), 34, 'backward')
    save_img('images/results/4_enlarge_dolphin_2', enlarge_dolphin_2)
    save_img('images/results/4_seam_enlarge_dolphin_2', seam_enlarge_dolphin_2)

    # bench with red seams to be removed by backward energy
    crop_bw_bench, seam_crop_bw_bench, _ = condense(img_bench, 50, 'backward')
    save_img('images/results/6_crop_bw_bench', crop_bw_bench)
    save_img('images/results/5_seam_crop_bw_bench', seam_crop_bw_bench)

    # bench with red seams removed by forward energy
    crop_fw_bench, seam_crop_fw_bench, _ = condense(img_bench, 50, 'forward')
    save_img('images/results/8_crop_fw_bench', crop_fw_bench)
    save_img('images/results/7_seam_crop_fw_bench', seam_crop_fw_bench)

    # stretched car with seams inserted using backward energy
    extend_bw_car, seam_extend_bw_car = extend(img_car, 50, 'backward')
    save_img('images/results/9_extend_bw_car', extend_bw_car)
    save_img('images/results/9_seam_extend_bw_car', seam_extend_bw_car)

    # stretched car with seams inserted using forward energy
    extend_fw_car, seam_extend_fw_car = extend(img_car, 50, 'forward')
    save_img('images/results/10_extend_fw_car', extend_fw_car)
    save_img('images/results/10_seam_extend_fw_car', seam_extend_fw_car)
