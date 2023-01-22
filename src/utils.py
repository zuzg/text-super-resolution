import cv2
import PIL
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from data import SRDataset


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
    img_plt = images_np.transpose(1,2,0)[:, :, ::-1]
    # print(img_plt.shape)
    plt.grid(False)
    plt.axis('off')
    plt.imshow(np.clip(img_plt, 0, 1))


def display_tensors(images_lists:list[tuple[torch.tensor]], title_list:list=None, 
                    suptitle:str=None, figsize=(15,5), display:bool=True) -> plt.figure:
    rows = len(images_lists)
    columns = len(images_lists[0])
    fig = plt.figure(figsize=figsize)
    titles = ["LR image", "HR image"]

    for i, images in enumerate(images_lists):
        for j, image in enumerate(images):

            if image.shape[1] == 32:
                image = image.add(1).div(2)

            image_np  = image.cpu().numpy()
            img_plt = image_np.transpose(1,2,0)[:, :, ::-1]
            fig.add_subplot(rows, columns, i*columns+j+1)
            plt.grid(False)
            plt.axis('off')
            if j%columns >= 2:
                if title_list is None or j%columns-2 >= len(title_list):
                    plt.title("SR image")
                else:
                    plt.title(title_list[j%columns-2])
            else:   
                plt.title(titles[j%3])
            plt.imshow(np.clip(img_plt, 0, 1))
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=20)
    if not display:
        plt.close(fig)
    fig.patch.set_facecolor('white')
    return fig


def get_prediction(LR_image:torch.tensor, model, display:bool=False) -> torch.tensor:
    LR_image = torch.unsqueeze(LR_image, dim=0)
    SR_image = model.forward(LR_image.float())
    SR_image = SR_image.detach()[0]
    if display:
        display_img_tensor(SR_image, rescale=True)
    return SR_image


def get_stats(HR_image:torch.tensor, SR_image:torch.tensor, data_range:int=2, display:bool=True):
    psnr_val = psnr(HR_image.cpu().numpy(), SR_image.cpu().numpy(), data_range=data_range)
    ssim_val = ssim(HR_image.cpu().numpy(), SR_image.cpu().numpy(), chanel_axis=0, data_range=data_range, win_size=3)
    if display:
        print(f'PSNR: {psnr_val:.3f}\nSSIM: {ssim_val:.3f}')
    return psnr_val, ssim_val


def get_exemplary_images(model, test_set:list[SRDataset], device:str, title:str=None, indices_list:list[int]=None, random:bool=False, n:int=10, figsize=(15, 15), display:bool=False):
    # NOTE: test_set here is an original test_set[difficulty_level]
    plt_rows = list()
    if indices_list is not None:
        iterate_list = indices_list
    elif random:
        iterate_list = random.sample(range(len(test_set)), n)
    else:
        iterate_list = range(n)

    for i in iterate_list:
        LR_image = test_set[i][0]
        HR_image = test_set[i][1]
        SR_image = get_prediction(LR_image.to(device), model)
        plt_rows.append([LR_image, HR_image, SR_image])

    return display_tensors(plt_rows, figsize=figsize, suptitle=title, display=display)


def evaluate_model(model, test_set:dict) -> tuple[float]:
    model.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
    keys = test_set.keys()
    avg_psnr = {key: 0 for key in keys}
    avg_ssim = {key: 0 for key in keys}
    for key in keys:
        for LR_image, HR_image in test_set[key]:
            SR_image = get_prediction(LR_image.to(device), model, display=False)
            psnr_val, ssim_val = get_stats(HR_image.to(device), SR_image, display=False)
            # print(psnr_val, ssim_val)
            avg_psnr[key] += psnr_val
            avg_ssim[key] += ssim_val
        avg_psnr[key] /= len(test_set[key])
        avg_ssim[key] /= len(test_set[key])
    return avg_psnr, avg_ssim
