import cv2
import PIL
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def imshow(a: np.ndarray):
    a = a.clip(0, 255).astype('uint8')
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(PIL.Image.fromarray(a))


def display_img_tensor(img:torch.tensor, rescale=False):
    if rescale:
        img = img.add(1).div(2)
    images_np  = img.cpu().numpy()
    img_plt = images_np.transpose(1,2,0)
    # print(img_plt.shape)
    plt.grid(False)
    plt.axis('off')
    plt.imshow(np.clip(img_plt, 0, 1))

def get_prediction(LR_image:torch.tensor, model, display:bool=True) -> torch.tensor:
    LR_image = torch.unsqueeze(LR_image, dim=0)
    SR_image = model.forward(LR_image.float())
    SR_image = SR_image.detach()[0]
    if display:
        display_img_tensor(SR_image, rescale=True)
    return SR_image

def get_stats(HR_image:torch.tensor, SR_image:torch.tensor, data_range:int=2, display:bool=True):
    psnr_val = psnr(HR_image.numpy(), SR_image.numpy(), data_range=data_range)
    ssim_val = ssim(HR_image.numpy(), SR_image.numpy(), chanel_axis=0, data_range=data_range, win_size=3)
    if display:
        print(f'PSNR :{psnr_val:.3f}\nSSIM: {ssim_val:.3f}')
    return psnr_val, ssim_val

def eval_model(model, test_set:dict) -> tuple[float]:
    model.eval()
    keys = test_set.keys()
    avg_psnr = {key: 0 for key in keys}
    avg_ssim = {key: 0 for key in keys}
    for key in keys:
        for LR_image, HR_image in test_set[key]:
            SR_image = get_prediction(LR_image, model, display=False)
            psnr_val, ssim_val = get_stats(HR_image, SR_image, display=False)
            # print(psnr_val, ssim_val)
            avg_psnr[key] += psnr_val
            avg_ssim[key] += ssim_val
        avg_psnr[key] /= len(test_set[key])
        avg_ssim[key] /= len(test_set[key])
    return avg_psnr, avg_ssim