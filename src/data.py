import cv2
import numpy as np
from src.utils import imshow
import matplotlib.pyplot as plt

# NOTE: we assume _img_HR.jpg and _img_HR.jpg suffices
def get_data_from_dir(dir_path:str, filenames:list[str], min_height:int=None) -> tuple[list[np.ndarray]]:
    images_LR_HR = list()

    for filename in filenames:
        # add an exception in case of no filename file
        imgLR = cv2.imread(dir_path+filename+'_img_LR.jpg', 1)
        imgHR = cv2.imread(dir_path+filename+'_img_HR.jpg', 1)
        if min_height is not None and (imgLR.shape[0] < min_height):
            continue
        images_LR_HR.append((imgLR, imgHR))

    return images_LR_HR

def show_LR_HR_images(imgLR, imgHR): # move to src/data?
    if imgHR.shape == imgLR.shape:
        imshow(cv2.resize(np.concatenate([imgLR, imgHR], 1), None, fx=2, fy=2))
    else:
        imshow(cv2.resize(imgLR, None, fx=2, fy=2))
        imshow(cv2.resize(imgHR, None, fx=2, fy=2))


def get_height_width_distribution(shapes_list:list[tuple[int]]):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    height_list = list(map(lambda x: x[0], shapes_list))
    width_list = list(map(lambda x: x[1], shapes_list))
    # print(height_list[:3])
    # print(width_list[:3])
    axs[0].hist(height_list, color='deeppink')
    axs[0].set_title("Images height distribution")
    axs[1].hist(width_list, color='deeppink')
    axs[1].set_title("Images width distribution")
    plt.show()