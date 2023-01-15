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

from src.models.srresnet import _NetG, _NetD
from src.cfg import read_config
# from src.data import SRDataset



#TODO - train_set type SRDataset once it is properly preprocessed and added to src.data
def sr_gan_perform_training(train_set, cfg:dict, pretrained:str=None, vgg_loss:bool=True, run_neptune:bool=True, save:str=None, verbose:bool=True):

    # read config parameters
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    step_lr = cfg["step_lr"]
    threads = cfg["threads"]

    global generative_model, discriminative_model, VGGmodel, step, device, run, NEPTUNE, api_token, beta
    NEPTUNE = run_neptune
    
    if NEPTUNE:
        api_token = read_config("cfg/tokens/api_token.yaml")["token"]
        run = neptune.init_run(project="super-girls/Super-Resolution", api_token=api_token)
        run["sys/tags"].add(["SRGAN"])
        run["params"] = cfg

    step=step_lr
    beta = cfg['beta']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cuda = torch.cuda.is_available()
    seed = 10
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True

    if train_set is None:
        print("No training set provided!")
        return 

    training_data_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batch_size)

    if vgg_loss:
        print("===> Runs with VGG model support (VGG loss applied)")
        netVGG = models.vgg19()
        netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))
        class _content_model(nn.Module):
            def __init__(self):
                super(_content_model, self).__init__()
                self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])
            def forward(self, x):
                out = self.feature(x)
                return out
        VGGmodel = _content_model()
        VGGmodel.eval()
        VGGmodel = VGGmodel.to(device)
    else:
        print("===> Runs without VGG (MSE loss applied)")

    generative_model = _NetG()
    discriminative_model = _NetD()
    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    generative_model.to(device)
    discriminative_model.to(device)
    content_loss_criterion.to(device)
    adversarial_loss_criterion.to(device)

    generative_model_total_params = sum(p.numel() for p in generative_model.parameters())
    discriminative_model_total_params = sum(p.numel() for p in discriminative_model.parameters())
    total_params = generative_model_total_params + discriminative_model_total_params

    # copy weights from a checkpoint (optional) TODO - add discriminator
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
    optimizer_d = optim.Adam(discriminative_model.parameters(), lr=lr)

    print("===> Training")
    for epoch in range(1, epochs + 1):
        train(training_data_loader, 
                optimizer_g, optimizer_d, 
                # generative_model, discriminative_model, 
                content_loss_criterion, adversarial_loss_criterion, 
                epoch, lr, vgg_loss, verbose)
    if NEPTUNE:
        run.stop()
    if save is not None:
        save_checkpoint(generative_model, epoch, save, total_params)


def train(training_data_loader, 
            optimizer_g, optimizer_d, 
            # generative_model, discriminative_model, 
            content_loss_criterion, adversarial_loss_criterion,
            epoch, lr, vgg_loss, verbose):

    lr = lr * (0.1 ** (epoch // step))
    # UPDATE LEARNING RATE
    for param_group in optimizer_g.param_groups:
        param_group["lr"] = lr
    for param_group in optimizer_d.param_groups:
        param_group["lr"] = lr

    generative_model.train()
    discriminative_model.train()

    if verbose:
        tepoch = tqdm(training_data_loader, unit="batch")
    else:
        tepoch = training_data_loader

    for input, target in tepoch:
        if verbose:
                tepoch.set_description(f"Epoch {epoch}")

        # print('Shapes: ', input.shape, target.shape)
        input.requires_grad = True
        target.requires_grad = False
   
        input, target = input.to(device), target.to(device)

        # GENERATOR
        output = generative_model(input.float())

        if vgg_loss:
            content_input = VGGmodel(output.float())
            content_target = VGGmodel(target.float()).detach()
            content_loss = content_loss_criterion(content_input, content_target)
        else:
            content_loss = content_loss_criterion(output.float(), target.float())
        # print('GENERATOR')
        # print('content_loss', content_loss)
        sr_discriminated = discriminative_model(output)
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + beta * adversarial_loss # beta coefficient to weight the adversarial loss in the perceptual loss
        # print('sr_discriminated', sr_discriminated)
        # print('adversarial_loss',adversarial_loss.item())
        # print('-----------------------------------')

        # optimize generator
        optimizer_g.zero_grad()
        if vgg_loss:
            VGGmodel.zero_grad()
        perceptual_loss.backward()
        optimizer_g.step()

        # DISCRIMINATOR
        hr_discriminated = discriminative_model(target.float())
        sr_discriminated = discriminative_model(output.detach())
        # print('DISCRIMINATOR')
        # print('hr_discriminated', hr_discriminated)
        # print('sr_discriminated', sr_discriminated)

        # print('discriminator losses')
        sr_predictions = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated))
        hr_predictions = adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))
        # print('sr_predictions', sr_predictions.item())
        # print('hr_predictions', hr_predictions.item())

        adversarial_loss = sr_predictions + hr_predictions

        # print('adversarial_loss',adversarial_loss.item())
        # optimize discriminator
        optimizer_d.zero_grad()
        adversarial_loss.backward()
        optimizer_d.step()

        if verbose:
            if vgg_loss:
                tepoch.set_postfix(content_loss_VGG=content_loss.item(), adversarial_loss=adversarial_loss.item(), perceptual_loss=perceptual_loss.item())
    
            else:
                tepoch.set_postfix(content_loss_MSE=content_loss.item(), adversarial_loss=adversarial_loss.item(), perceptual_loss=perceptual_loss.item())
    if vgg_loss and NEPTUNE:
        run["train/content_loss_VGG"].append(content_loss.item())
        run["train/adversarial_loss"].append(adversarial_loss.item())
        run["train/perceptual_loss"].append(perceptual_loss.item())
    elif NEPTUNE:
        run["train/content_loss_MSE"].append(content_loss.item())
        run["train/adversarial_loss"].append(adversarial_loss.item())
        run["train/perceptual_loss"].append(perceptual_loss.item())


def save_checkpoint(model, epoch, save_name, params):
    model_out_path = "checkpoint/" + f"model_{save_name}.pth"
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)
    print(f"Model saved to {model_out_path}")

    # if new_model is not None:
    #     model = neptune.init_model(key=f"SRRN{save_name}", project="super-girls/Super-Resolution", api_token=api_token)
    #     model["sys/tags"].add(["SRResNet"])
    #     model["total_params"] = params
    #     model_id = model["sys/id"].fetch()
    if NEPTUNE:
        model_version = neptune.init_model_version(model="SR-SRRN6", project="super-girls/Super-Resolution", api_token=api_token)
        model_version["weights"].upload(f"{model_out_path}")
        # model.stop()
        model_version.stop()

if __name__ == "__main__":
    sr_gan_perform_training()
