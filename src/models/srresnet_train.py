import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
from tqdm import tqdm
from src.models.srresnet import _NetG, _NetD

#TODO - train_set type
def sr_resnet_perform_training(train_set = None, batch_size:int=16, epochs:int=500,
                    lr:float=1e-4, step_lr:int=200, threads:int=0, 
                    pretrained:str=None, vgg_loss:bool=True, save:bool=False, verbose:bool=True):

    global generative_model, adversarial_model, VGGmodel, step, device

    step=step_lr
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cuda = torch.cuda.is_available()
    seed = 10 # for reproducibility
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    # Enable the inbuilt cudnn auto-tuner -> faster runtime
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
        VGGmodel = VGGmodel.to(device)
    else:
        print("===> Runs without VGG (MSE loss applied)")

    generative_model = _NetG()
    adversarial_model = _NetD()
    content_loss_criterion = nn.MSELoss()
    generative_model.to(device)
    content_loss_criterion.to(device)

    # copy weights from a checkpoint (optional)
    if pretrained:
        if os.path.isfile(pretrained):
            print("=> loading generative model '{}'".format(pretrained))
            weights = torch.load(pretrained)
            generative_model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no generative model found at '{}'".format(pretrained))
            return 

    # print("===> Setting Optimizer")
    optimizer = optim.Adam(generative_model.parameters(), lr=lr)

    print("===> Training")
    for epoch in range(1, epochs + 1):
        train(training_data_loader, optimizer, generative_model, content_loss_criterion, epoch, lr, vgg_loss, verbose)
        if save and epoch%10==0:
            save_checkpoint(generative_model, epoch)


def train(training_data_loader, optimizer, generative_model, content_loss_criterion, epoch, lr, vgg_loss, verbose):

    lr = lr * (0.1 ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    generative_model.train()

    if verbose:
        tepoch = tqdm(training_data_loader, unit="batch")
    else:
        tepoch = training_data_loader

    for input, target in tepoch:
        if verbose:
                tepoch.set_description(f"Epoch {epoch}")

        # print('Shapes: ',, input.shape, target.shape)
        input.requires_grad = True
        target.requires_grad = False
   
        input, target = input.to(device), target.to(device)

        output = generative_model(input.float())
        content_loss_MSE = content_loss_criterion(output.float(), target.float())

        if vgg_loss:
            content_input = VGGmodel(output.float())
            content_target = VGGmodel(target.float())
            content_target = content_target.detach()
            content_loss_VGG = content_loss_criterion(content_input, content_target)
        
        optimizer.zero_grad()

        if vgg_loss:
            VGGmodel.zero_grad()
            content_loss_VGG.backward(retain_graph=True)
        
        else:
            content_loss_MSE.backward()

        optimizer.step()

        if verbose:
            if vgg_loss:
                tepoch.set_postfix(content_loss_VGG=content_loss_VGG.item(), content_loss_MSE=content_loss_MSE.item())
            else:
                tepoch.set_postfix(content_loss_MSE=content_loss_MSE.item())

        # if iteration%len(training_data_loader) == 0:
        #     if vgg_loss:
        #         print("===> Epoch[{}]({}/{}): Loss: {:.5} Content_loss {:.5}".format(epoch, iteration, len(training_data_loader), loss.data, content_loss.data))
        #     else:
        #         print("===> Epoch[{}]({}/{}): Loss: {:.5}".format(epoch, iteration, len(training_data_loader), loss.data))

def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)
    print("Model saved to {}".format(model_out_path))

if __name__ == "__main__":
    sr_resnet_perform_training()