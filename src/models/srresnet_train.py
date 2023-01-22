import os
import sys
if '../' not in sys.path:
    sys.path.append('../')
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
from tqdm import tqdm
import neptune.new as neptune

from models.srresnet import _NetG
from cfg import read_config
from utils import evaluate_model, get_exemplary_images
from data import SRDataset


def sr_resnet_perform_training(train_set: SRDataset, cfg: dict, generative_model=_NetG(),
                               test_set: dict[SRDataset] = None, pretrained: str = None,
                               vgg_loss: bool = True, run_neptune: bool = True,
                               save: str = None, verbose: bool = True):
    # read config parameters
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    step_lr = cfg["step_lr"]
    threads = cfg["threads"]

    global VGGmodel, step, device, run, NEPTUNE, api_token, beta
    NEPTUNE = run_neptune

    if NEPTUNE:
        api_token = read_config("cfg/tokens/api_token.yaml")["token"]
        run = neptune.init_run(
            project="super-girls/Super-Resolution", api_token=api_token)
        if isinstance(generative_model, _NetG):
            run["sys/tags"].add(["SRResNet"])
        else:
            run["sys/tags"].add(["E-Generator"])
        run["params"] = cfg

    step = step_lr
    beta = cfg['beta']
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    cuda = torch.cuda.is_available()
    seed = 10
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True

    if train_set is None:
        print("No training set provided!")
        return

    training_data_loader = DataLoader(
        dataset=train_set, num_workers=threads, batch_size=batch_size)

    if vgg_loss:
        print("===> Runs with VGG model support (VGG loss applied)")
        netVGG = models.vgg19()
        netVGG.load_state_dict(model_zoo.load_url(
            'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))

        class _content_model(nn.Module):
            def __init__(self):
                super(_content_model, self).__init__()
                self.feature = nn.Sequential(
                    *list(netVGG.features.children())[:-1])

            def forward(self, x):
                out = self.feature(x)
                return out
        VGGmodel = _content_model()
        VGGmodel = VGGmodel.to(device)
    else:
        print("===> Runs without VGG (MSE loss applied)")

    content_loss_criterion = nn.MSELoss()
    # content_loss_criterion = nn.MSELoss(reduction='none')

    generative_model.to(device)
    content_loss_criterion.to(device)

    generative_model_total_params = sum(
        p.numel() for p in generative_model.parameters())
    total_params = generative_model_total_params

    # copy weights from a checkpoint (optional)
    if pretrained:
        if os.path.isfile(pretrained):
            print("=> loading generative model '{}'".format(pretrained))
            weights = torch.load(pretrained)
            generative_model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no generative model found at '{}'".format(pretrained))
            return

    print("===> Setting Optimizers")
    optimizer_g = optim.Adam(generative_model.parameters(), lr=lr)

    print("===> Training")
    for epoch in range(1, epochs + 1):
        train(generative_model, training_data_loader,
              optimizer_g,
              content_loss_criterion,
              epoch, lr, vgg_loss, verbose)
        if test_set is not None:
            avg_psnr, avg_ssim = evaluate_model(generative_model, test_set)
            if NEPTUNE:
                for mode in avg_psnr.keys():
                    run[f"eval/{mode}/psnr_avg"].append(avg_psnr[mode])
                    run[f"eval/{mode}/ssim_avg"].append(avg_ssim[mode])
                    if epoch%10 == 0:
                        plt_fig = get_exemplary_images(generative_model, test_set[mode], device, title=f'{mode.title()} test set [EPOCH {epoch}]')
                        run[f"eval/{mode}/images"].append(plt_fig)
        if save is not None and epoch % 50 == 0:
            save_checkpoint(generative_model, epoch, save, total_params)
    if NEPTUNE:
        run.stop()
    if save is not None:
        save_checkpoint(generative_model, epoch, save, total_params)


def train(generative_model, training_data_loader,
          optimizer_g,
          content_loss_criterion,
          epoch, lr, vgg_loss, verbose):

    lr = lr * (0.1 ** (epoch // step))
    # UPDATE LEARNING RATE
    for param_group in optimizer_g.param_groups:
        param_group["lr"] = lr

    generative_model.train()

    if verbose:
        tepoch = tqdm(training_data_loader, unit="batch")
    else:
        tepoch = training_data_loader

    for input, target in tepoch:
        if verbose:
            tepoch.set_description(f"Epoch {epoch}")

        input.requires_grad = True
        target.requires_grad = False

        input, target = input.to(device), target.to(device)

        # GENERATOR
        output = generative_model(input.float())
        loss = content_loss_criterion(output.float(), target.float())

        if vgg_loss:
            content_input = VGGmodel(output.float())
            content_target = VGGmodel(target.float()).detach()
            content_loss = content_loss_criterion(
                content_input, content_target)

        # optimize generator
        optimizer_g.zero_grad()
        if vgg_loss:
            VGGmodel.zero_grad()
            content_loss.backward(retain_graph=True)

        loss.backward()
        optimizer_g.step()

        if verbose:
            if vgg_loss:
                tepoch.set_postfix(MSE_loss=loss.item(),
                                   VGG_loss=content_loss.item())

            else:
                tepoch.set_postfix(MSE_loss=loss.item())
    if vgg_loss and NEPTUNE:
        run["train/VGG_loss"].append(content_loss.item())
        run["train/MSE_loss"].append(loss.item())

    elif NEPTUNE:
        run["train/MSE_loss"].append(loss.item())


def save_checkpoint(model, epoch, save_name, params):
    model_out_path = "checkpoint/" + f"model_{save_name}.pth"
    state = {"epoch": epoch, "model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)
    print(f"Model saved to {model_out_path}")

    # TODO: run this when we have final model
    # model_nept = neptune.init_model(key=f"E", project="super-girls/Super-Resolution", api_token=api_token)
    # model_nept["sys/tags"].add(["Egenerator"])
    # model_nept["total_params"] = params

    if NEPTUNE:
        if isinstance(model, _NetG):
            model_name = "SR-RESNET"
        else:
            model_name = "SR-E"
        model_version = neptune.init_model_version(
            model=model_name, project="super-girls/Super-Resolution", api_token=api_token)
        model_version["weights"].upload(f"{model_out_path}")
        model_version.stop()
    # model_nept.stop()


if __name__ == "__main__":
    sr_resnet_perform_training()
