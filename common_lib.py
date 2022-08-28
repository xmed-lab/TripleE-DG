import torch
import tqdm
import torchvision.transforms as transforms
from aug_lib import TrivialAugment

def get_train_standard_transformers():

    img_tr = [transforms.RandomResizedCrop((224,224), (0.8, 1.0))]
    img_tr.append(transforms.RandomHorizontalFlip(0.5))
    img_tr.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                                             hue=min(0.5, 0.4)))

    transform_train = transforms.Compose(img_tr)

    tile_tr = []
    tile_tr.append(transforms.RandomGrayscale(0.1))
    tile_tr = tile_tr + [transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transform_train, transforms.Compose(tile_tr)


def get_train_ESAugSole_transformers():

    TAaugment = TrivialAugment()
    img_tr = [transforms.RandomResizedCrop((224,224), (0.8, 1.0))]

    # remove other augs and 14 augmentation list
    img_tr.append(TAaugment)
    transform_train = transforms.Compose(img_tr)

    tile_tr = []
    tile_tr.append(transforms.RandomGrayscale(0.1))
    tile_tr = tile_tr + [transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transform_train, transforms.Compose(tile_tr)


def get_val_transformer():
    img_tr = [transforms.Resize((224,224)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)

def get_optim_and_scheduler(network, epochs, lr, train_all=True):
    if train_all:
        params = network.parameters()
    else:
        params = network.get_params(lr)
    optimizer = torch.optim.SGD(params, weight_decay=2e-5, momentum=.9, nesterov=False, lr=lr)
    step_size = int(epochs * 0.4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    return optimizer, scheduler

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)