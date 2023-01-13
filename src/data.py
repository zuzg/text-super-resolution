import cv2
import numpy as np

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