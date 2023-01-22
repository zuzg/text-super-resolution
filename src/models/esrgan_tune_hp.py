import sys
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
import optuna
from optuna.trial import TrialState

from utils import evaluate_model
from data import get_train_test
from models.esrgan import NetG_E, NetD_E


def objective(trial):
    batch_size = trial.suggest_int("batch_size", 8, 32, step=8)
    epochs = 10
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    step_lr = 200
    threads = 0
    momentum = trial.suggest_uniform("momentum", 0.1, 0.9)

    global VGGmodel, step, device, beta
    step = step_lr
    beta = 0.01
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    generator = NetG_E() # option: use pretrained
    generator.to(device)
    discriminator = NetD_E()
    discriminator.to(device)

    weights = torch.load('checkpoint/model_e.pth', map_location=torch.device(device))
    generator.load_state_dict(weights['model'].state_dict())

    train_set, test_set = get_train_test()
    training_data_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batch_size)

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

    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    generator.to(device)
    discriminator.to(device)
    content_loss_criterion.to(device)
    adversarial_loss_criterion.to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, )
    optimizer_d = optim.SGD(discriminator.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(generator, discriminator,
              training_data_loader,
              optimizer_g, optimizer_d,
              content_loss_criterion, adversarial_loss_criterion,
              epoch, lr)
        avg_psnr, avg_ssim = evaluate_model(generator, test_set)
        trial.report(avg_psnr['medium'], epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_psnr['medium']


def train(generator, discriminator,
          training_data_loader,
          optimizer_g, optimizer_d,
          content_loss_criterion, adversarial_loss_criterion,
          epoch, lr):

    lr = lr * (0.1 ** (epoch // step))
    for param_group in optimizer_g.param_groups:
        param_group["lr"] = lr
    for param_group in optimizer_d.param_groups:
        param_group["lr"] = lr

    generator.train()
    discriminator.train()
    tepoch = training_data_loader

    for input, target in tepoch:
        input.requires_grad = True
        target.requires_grad = False
        input, target = input.to(device), target.to(device)
        output = generator(input.float())

        content_input = VGGmodel(output.float())
        content_target = VGGmodel(target.float()).detach()
        content_loss = content_loss_criterion(
            content_input, content_target)

        sr_discriminated = discriminator(output)
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + beta * adversarial_loss

        optimizer_g.zero_grad()
        VGGmodel.zero_grad()
        perceptual_loss.backward()
        optimizer_g.step()

        hr_discriminated = discriminator(target.float())
        sr_discriminated = discriminator(output.detach())

        sr_predictions = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated))
        hr_predictions = adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))
        adversarial_loss = sr_predictions + hr_predictions


def tune_hp(n_trials=10):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True) # timeout=600, 

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return study
