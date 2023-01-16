import cv2
import PIL
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import torch

def imshow(a: np.ndarray):
    a = a.clip(0, 255).astype('uint8')
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(PIL.Image.fromarray(a))


def display_img_tensor(img, rescale=False):
    if rescale:
        img = img.add(1).div(2)
    images_np  = img.cpu().numpy()
    img_plt = images_np.transpose(1,2,0)
    # print(img_plt.shape)
    plt.grid(False)
    plt.axis('off')
    plt.imshow(np.clip(img_plt, 0, 1))

def show_prediction(LR_image, model):
    LR_image = torch.unsqueeze(LR_image, dim=0)
    SR_image = model.forward(LR_image.float())
    display_img_tensor(SR_image.detach()[0], rescale=True)