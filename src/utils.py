import cv2
import PIL
import numpy as np
from IPython.display import display

def imshow(a:np.ndarray):
    a = a.clip(0, 255).astype('uint8')
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(PIL.Image.fromarray(a))

# NOTE: we assume _img_HR.jpg and _img_HR.jpg suffices
def get_data_from_dir(dir_path:str, filenames:list[str]) -> tuple[list[np.ndarray]]:
    images_LR_HR = list()

    for filename in filenames:
        # add an exception in case of no filename file
        imgLR = cv2.imread(dir_path+filename+'_img_LR.jpg', 1) 
        imgHR = cv2.imread(dir_path+filename+'_img_HR.jpg', 1)
        images_LR_HR.append((imgLR, imgHR))

    return images_LR_HR