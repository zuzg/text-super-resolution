import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from src.utils import imshow

# TODO replace this with final one
class SRDataset(Dataset):
    def __init__(self, images, crop_size=None, normalize=True): 
        self.normalize = normalize
        self.crop_size = crop_size
        self.images = images

    def __len__(self):
        return len(self.images)

    def preprocess_image(self, image):
        if self.crop_size is not None:
            hr_cropped = cv2.resize(image[0], self.crop_size)
            lr_cropped = cv2.resize(image[1], (24, 24))
        # TODO patching?
        if self.normalize:
            lr_norm = lr_cropped / 255
            hr_norm = hr_cropped / 255 # [0, 1]

            return torch.tensor(lr_norm).swapaxes(1,2).swapaxes(0,1), torch.tensor(hr_norm).swapaxes(1,2).swapaxes(0,1)
        return torch.tensor(image[0]).swapaxes(1,2).swapaxes(0,1), torch.tensor(image[1]).swapaxes(1,2).swapaxes(0,1)

    def __getitem__(self, index):        
        image = self.images[index]         
        return self.preprocess_image(image)


# NOTE: we assume _img_HR.jpg and _img_HR.jpg suffices
def get_data_from_dir(dir_path: str, filenames: list[str], min_height: int = None) -> tuple[list[np.ndarray]]:
    images_LR_HR = list()

    for filename in filenames:
        # add an exception in case of no filename file
        imgLR = cv2.imread(dir_path+filename+'_img_LR.jpg', 1)
        imgHR = cv2.imread(dir_path+filename+'_img_HR.jpg', 1)
        if min_height is not None and (imgLR.shape[0] < min_height):
            continue
        images_LR_HR.append((imgLR, imgHR))

    return images_LR_HR


def show_LR_HR_images(imgLR, imgHR):
    if imgHR.shape == imgLR.shape:
        imshow(cv2.resize(np.concatenate([imgLR, imgHR], 1), None, fx=2, fy=2))
    else:
        imshow(cv2.resize(imgLR, None, fx=2, fy=2))
        imshow(cv2.resize(imgHR, None, fx=2, fy=2))


def get_height_width_distribution(shapes_list: list[tuple[int]]):
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


def transform_(path):
    img = PIL.Image.open(path)
    img = img.resize((64, 16), PIL.Image.BICUBIC)
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor
