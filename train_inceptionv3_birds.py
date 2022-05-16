from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf
import timm
from timm.loss import LabelSmoothingCrossEntropy
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, GaussianBlur, RandomRotation, ToTensor
from tqdm import tqdm
import wandb


parser = ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--config', type=str)
parser.add_argument('--run_name', type=str)
parser.add_argument('--api_key', type=str)
parser.add_argument('--seed', type=int, default=31415926)


def get_dataloaders(data_dir, config):
    transforms = Compose((
        ToTensor(),
        # Resize(224, 224),
        GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        RandomRotation((0, 180))
    ))
    train_data = ImageFolder(data_dir / 'train', transform=transforms)
    eval_data = ImageFolder(data_dir / 'test', transform=transforms)
    tsampler = RandomSampler(train_data)
    esampler = RandomSampler(eval_data)
    return (
        DataLoader(train_data, batch_size=config.train.batch_size, sampler=tsampler),
        DataLoader(eval_data, batch_size=config.train.batch_size, sampler=esampler)
    )


def eval(model, data, device, config):
    model.eval()
    loss_fn = LabelSmoothingCrossEntropy(0.0)
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(data):
            images, target = batch        
            losses.append(loss_fn(model(images.to(device)), target.to(device)).unsqueeze(0))
    loss = torch.mean(torch.cat(losses))
    wandb.log({'loss/test': loss})
    

def train(model, data_dir, config):
    device = 'cuda:0'
    model = model.to(device)
    loss_fn = LabelSmoothingCrossEntropy(config.train.label_smoothing)
    optim = torch.optim.Adam(model.parameters())
    train_data, eval_data = get_dataloaders(data_dir, config)
    # TODO 1. LR Scheduler
    # TODO 2. Distributed Data Parallel
    # TODO 5. Model checkpointing
    for epoch in range(config.train.num_epochs):
        print('#####################################################')
        print(f'######Epoch {epoch}')
        print('####################################################')
        for i, batch in tqdm(enumerate(train_data)):
            images, target = batch
            predictions = model(images.to(device))
            target = target.to(device)
            loss = loss_fn(predictions, target.to(device))
            loss.backward()
            optim.step()
            wandb.log({'loss/train': loss})
            
            if i % config.train.eval_interval:
                predictions.detach(), target.detach()
                predictions.to('cpu'), images.to('cpu')
                try:
                    eval(model, eval_data, device, config)
                finally:
                    model.train()
            
            
if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    config = OmegaConf.load(args.config)
    model = timm.create_model('inception_v3', pretrained=False, in_chans=3, num_classes=config.data.num_classes)
    wandb.login(key=args.api_key)
    wandb.init(
        entity='dstander',
        project='multimodal-fid',
        name=args.run_name
    )
    wandb.config.update(OmegaConf.to_container(config))
    # wandb.config.update(args)
    train(model, Path(args.data_dir), config)
