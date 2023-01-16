import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src.mdb_to_jpg import mdb_to_jpg
from src.utils import imshow

class SRDataset(Dataset):
    def __init__(self, images, crop=True, normalize=True): 
        self.normalize = normalize
        self.crop = crop
        self.images = images

    def __len__(self):
        return len(self.images)

    def preprocess_image(self, image):
        lr = image[0]
        hr = image[1]
        if self.crop:
            lr = cv2.resize(lr, (64, 16))
            hr = cv2.resize(hr, (128, 32))
        if self.normalize:

            lr = lr / 255 # [0; 1]
            # lr = lr.astype(np.float32)
            # lr = (lr - 127.5) / 127.5 # [-1; 1]

            hr = hr.astype(np.float32)
            hr = (hr - 127.5) / 127.5 # [-1; 1]

        return torch.tensor(lr).swapaxes(1,2).swapaxes(0,1), torch.tensor(hr).swapaxes(1,2).swapaxes(0,1)

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

def load_tests_sets(difficulty_levels:list[str]= ['easy', 'medium', 'hard'], n_test:int=None, convert_mdb:bool=True) -> dict[SRDataset]:
    test_data_dict = dict()

    for difficulty in difficulty_levels:
        print(str.upper(difficulty))
        test_data_path = f'data/TextZoom/test_img/{difficulty}/'
        if convert_mdb:
            lmdb_file = f'data/TextZoom/test/{difficulty}'
            n = mdb_to_jpg(test_data_path, lmdb_file)
        if n_test is None:
            test_img_data = get_data_from_dir(test_data_path, [str(i) for i in range(1, n+1)])
        else:
            test_img_data = get_data_from_dir(test_data_path, [str(i) for i in range(1, n_test+1)])
        test_img_data_processed = SRDataset(test_img_data)
        test_data_dict[difficulty] = test_img_data_processed
    return test_data_dict

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
