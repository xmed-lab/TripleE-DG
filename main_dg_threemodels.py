#!/usr/bin/env python
import argparse
import os
import torch
import torch.nn as nn
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from tensorboardX import SummaryWriter
from os.path import join
import matplotlib

matplotlib.use('pdf')
from tqdm import tqdm
from models.resnet_eis import resnet50, resnet18
from domain.losses import SupConLoss

from domain.losses import kNN, multi_kNN
import torch.nn.functional as F
from torch.utils.data import random_split
from common_lib import get_train_standard_transformers, get_train_ESAugSole_transformers, get_val_transformer, \
    get_optim_and_scheduler, get_lr, save_checkpoint, AverageMeter

from stylize_data import stylize_image
import net
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

torch.set_num_threads(6)
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', type=str, default="resnet18")
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--expname', default='experiment', type=str)

#  options for dg
parser.add_argument("--result", default="exp/skin/skin_moco_v1/", type=str)
parser.add_argument("--target", nargs='+')
parser.add_argument("--evaluate", action="store_true")

parser.add_argument("--ratio", default=1.0, type=float)
parser.add_argument("--baug", default=1, type=int)
parser.add_argument("--times", default=1, type=int)

parser.add_argument("--ESAugF", action="store_true")
parser.add_argument("--ESAugS", action="store_true")
parser.add_argument("--classes", default=65, type=int)

parser.add_argument("--resume_1", default="exp/skin/skin_moco_v1/", type=str)
parser.add_argument("--resume_2", default="exp/skin/skin_moco_v1/", type=str)
parser.add_argument("--resume_3", default="exp/skin/skin_moco_v1/", type=str)

args = parser.parse_args()


if args.classes == 65:
    main_path = "../../FACT-main-second/DATASET/OfficeHome/"
    filelists = "office_txt_lists"
    domain_list = ["Clipart", "Art", "Product", "Real_World"]
elif args.classes == 7:
    main_path = "../../FACT-main-second/"
    filelists = "txt_lists"
    domain_list = ["art_painting", "cartoon", "sketch", "photo"]
elif args.classes == 345:
    main_path = "/home/eexmli/193dataset/"
    filelists = "domainbed_txt_lists"
    domain_list = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
else:
    exit(0)




## Global para.
source_domain = list(set(domain_list) - set(args.target))

# load stylize model
decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()
decoder.load_state_dict(torch.load('pre_models/decoder.pth'))
vgg.load_state_dict(torch.load('pre_models/vgg_normalised.pth'))
vgg = nn.Sequential(*list(vgg.children())[:31])
vgg.cuda()
decoder.cuda()



def main():

    main_worker(args)


def main_worker(args):


    # create model
    if args.arch == "resnet18":
        model_1 = resnet18(classes=args.classes, args=args)
        model_2 = resnet18(classes=args.classes, args=args)
        model_3 = resnet18(classes=args.classes, args=args)
        print("=> creating model '{}'".format("resnet18"))
    else:
        model_1 = resnet50(classes=args.classes, args=args)
        model_2 = resnet50(classes=args.classes, args=args)
        model_3 = resnet50(classes=args.classes, args=args)
        print("=> creating model '{}'".format("resnet50"))

    model_1 = nn.DataParallel(model_1).cuda()
    model_2 = nn.DataParallel(model_2).cuda()
    model_3 = nn.DataParallel(model_3).cuda()

    criterion = SupConLoss().cuda()
    ce_criterion = nn.CrossEntropyLoss()

    optimizer_1, scheduler_1 = get_optim_and_scheduler(model_1, args.epochs, args.lr)
    optimizer_2, scheduler_2 = get_optim_and_scheduler(model_2, args.epochs, args.lr)
    optimizer_3, scheduler_3 = get_optim_and_scheduler(model_3, args.epochs, args.lr)

    cudnn.benchmark = True

    #  get augmentations
    if args.ESAugS:
        print("Use ESAug with Style")
        img_transformer, tile_transformer = get_train_ESAugSole_transformers()  ## ESAug
    elif args.ESAugF:
        print("Use ESAug with Fourier")
        img_transformer, tile_transformer = get_train_ESAugSole_transformers()  ## ESAug
    else:
        print("Use StanardAug")
        img_transformer, tile_transformer = get_train_standard_transformers()

    #  load dataset
    datasets_train, datasets_val, datasets_train_val = [], [], []
    from datasets.dg_dataset import JigsawDataset, BaselineDataset, ConcatDataset, _dataset_info, get_random_subset

    for dname in source_domain:
        if args.classes == 345:
            name_train, labels_train = _dataset_info(join(args.data, 'data', filelists, '%s_train.txt' % dname))
            name_train, name_val, labels_train, labels_val = get_random_subset(name_train, labels_train, 0.1)
        else:
            name_train, labels_train = _dataset_info(join(args.data, 'data', filelists, '%s_train.txt' % dname))
            name_val, labels_val = _dataset_info(join(args.data, 'data', filelists, '%s_val.txt' % dname))

        # change path
        name_train = [main_path + item for item in name_train]
        name_val = [main_path + item for item in name_val]
        train_dataset = JigsawDataset(name_train, labels_train, patches=False, img_transformer=img_transformer,
                                      tile_transformer=tile_transformer, jig_classes=30, bias_whole_image=0.9,
                                      args=args, cropsize=224)
        datasets_train.append(train_dataset)

        train_dataset = BaselineDataset(name_val, labels_val, img_transformer=get_val_transformer(),
                                        two_transform=False)
        datasets_val.append(train_dataset)

        train_val_dataset = BaselineDataset(name_train, labels_train, img_transformer=get_val_transformer(),
                                            two_transform=False)
        datasets_train_val.append(train_val_dataset)

    datasets_train = ConcatDataset(datasets_train)
    datasets_val = ConcatDataset(datasets_val)
    datasets_train_val = ConcatDataset(datasets_train_val)

    train_loader = torch.utils.data.DataLoader(datasets_train, batch_size=args.batch_size, shuffle=True, num_workers=12,
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(datasets_val, batch_size=args.batch_size, shuffle=False, num_workers=12,
                                             pin_memory=True, drop_last=False)
    print("Dataset size: train %d, val %d" % (len(train_loader.dataset), len(val_loader.dataset)))

    name_test, labels_test = _dataset_info(join(args.data, 'data', filelists, '%s_test.txt' % args.target[0]))
    name_test = [main_path + item for item in name_test]

    test_dataset = BaselineDataset(name_test, labels_test, img_transformer=get_val_transformer())
    test_loader = torch.utils.data.DataLoader(ConcatDataset([test_dataset]), batch_size=args.batch_size, shuffle=False,
                                              num_workers=12, pin_memory=True, drop_last=False)

    print("Dataset size: test %d" % (len(test_loader.dataset)))

    if args.evaluate:
        if os.path.isfile(args.resume_1):
            print("=> loading checkpoint '{}'".format(args.resume_1))
            checkpoint = torch.load(args.resume_1)
            model_1.load_state_dict(checkpoint['state_dict'])

        if os.path.isfile(args.resume_2):
            print("=> loading checkpoint '{}'".format(args.resume_2))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(args.resume_2)
            model_2.load_state_dict(checkpoint['state_dict'])

        if os.path.isfile(args.resume_3):
            print("=> loading checkpoint '{}'".format(args.resume_3))
            checkpoint = torch.load(args.resume_3)
            model_3.load_state_dict(checkpoint['state_dict'])

        # test_acc_1 = multi_kNN(10, model_1, model_1, model_2, model_3, test_loader, datasets_train_val, args, args.classes)
        test_acc_1 = testmodel(model_1, model_2, model_3, test_loader)
        print ("acc", test_acc_1)
        exit(0)

    train_len = len(datasets_train)
    sub1_datasets_train, sub2_datasets_train, sub3_datasets_train = random_split(
        datasets_train, lengths = [int(train_len/3), int(train_len/3), train_len - int(train_len/3) * 2], generator=torch.Generator().manual_seed(10))

    sub_train_loader_1 = torch.utils.data.DataLoader(sub1_datasets_train, batch_size=args.batch_size, shuffle=True, num_workers=12,
                                         pin_memory=True, drop_last=True)
    sub_train_loader_2 = torch.utils.data.DataLoader(sub2_datasets_train, batch_size=args.batch_size, shuffle=True, num_workers=12,
                                         pin_memory=True, drop_last=True)
    sub_train_loader_3 = torch.utils.data.DataLoader(sub3_datasets_train, batch_size=args.batch_size, shuffle=True, num_workers=12,
                                         pin_memory=True, drop_last=True)
    print("SubDataset size: subdata1 %d, subdata2 %d, subdata3 %d" % (
        len(sub_train_loader_1.dataset), len(sub_train_loader_2.dataset), len(sub_train_loader_3.dataset)))

    # mkdir result folder and tensorboard
    os.makedirs(args.result, exist_ok=True)
    writer = SummaryWriter("runs/" + str(args.result.split("/")[-1]))
    writer.add_text('Text', str(args))

    best_acc = 0
    best_test_acc = 0
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if epoch >= 10:
            train_len = len(datasets_train)
            sub1_datasets_train, sub2_datasets_train, sub3_datasets_train = random_split(
                datasets_train, lengths=[int(train_len / 3), int(train_len / 3), train_len - int(train_len / 3) * 2])

            sub_train_loader_1 = torch.utils.data.DataLoader(sub1_datasets_train, batch_size=args.batch_size,
                                                             shuffle=True, num_workers=12,
                                                             pin_memory=True, drop_last=True)
            sub_train_loader_2 = torch.utils.data.DataLoader(sub2_datasets_train, batch_size=args.batch_size,
                                                             shuffle=True, num_workers=12,
                                                             pin_memory=True, drop_last=True)
            sub_train_loader_3 = torch.utils.data.DataLoader(sub3_datasets_train, batch_size=args.batch_size,
                                                             shuffle=True, num_workers=12,
                                                             pin_memory=True, drop_last=True)
            print("SubDataset size: subdata1 %d, subdata2 %d, subdata3 %d" % (
                len(sub_train_loader_1.dataset), len(sub_train_loader_2.dataset), len(sub_train_loader_3.dataset)))

        sub_loss_1 = train(sub_train_loader_1, model_1, criterion, ce_criterion, optimizer_1, args)
        sub_loss_2 = train(sub_train_loader_2, model_2, criterion, ce_criterion, optimizer_2, args)
        sub_loss_3 = train(sub_train_loader_3, model_3, criterion, ce_criterion, optimizer_3, args)

        loss = sub_loss_1 + sub_loss_2 + sub_loss_3
        writer.add_scalar("train_loss", loss, epoch + 1)


        scheduler_1.step()
        scheduler_2.step()
        scheduler_3.step()
        # writer.add_scalar('lr', get_lr(optimizer_1), epoch + 1)

        # val
        # acc = multi_kNN(500, model, sub_model_1, sub_model_2, sub_model_3, val_loader, datasets_train_val, args, 10)
        acc = testmodel(model_1, model_2, model_3, val_loader)
        writer.add_scalar("val_acc", acc, epoch + 1)

        print(
            f"[ Val | {epoch + 1:03d}/{args.epochs:03d} ] val_acc = {acc:.5f}")
        # test_acc = multi_kNN(500, model, sub_model_1, sub_model_2, sub_model_3, test_loader, datasets_train_val, args, 10)
        test_acc = testmodel(model_1, model_2, model_3, test_loader)
        writer.add_scalar("test_acc", test_acc, epoch + 1)
        """@nni.report_intermediate_result(test_acc.item())"""


        print(
            f"[ Test | {epoch + 1:03d}/{args.epochs:03d} ] test_acc = {test_acc:.5f}")

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model_1.state_dict(),
            'optimizer': optimizer_1.state_dict(),
        }, is_best=False, filename=args.result + '/sub_model_1_last.pth.tar')

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model_2.state_dict(),
            'optimizer': optimizer_2.state_dict(),
        }, is_best=False, filename=args.result + '/sub_model_2_last.pth.tar')

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model_3.state_dict(),
            'optimizer': optimizer_3.state_dict(),
        }, is_best=False, filename=args.result + '/sub_model_3_last.pth.tar')

        if acc >= best_acc:
            best_acc = acc
            best_test_acc = test_acc
            best_epoch = epoch
            if epoch >= 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model_1.state_dict(),
                    'optimizer': optimizer_1.state_dict(),
                }, is_best=False, filename=args.result + '/sub_model_1_best.pth.tar')

                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model_2.state_dict(),
                    'optimizer': optimizer_2.state_dict(),
                }, is_best=False, filename=args.result + '/sub_model_2_best.pth.tar')

                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model_3.state_dict(),
                    'optimizer': optimizer_3.state_dict(),
                }, is_best=False, filename=args.result + '/sub_model_3_best.pth.tar')

    print("best acc for " + args.target[0], best_test_acc, best_epoch)
    """@nni.report_final_result(best_test_acc.item())"""

def train(train_loader, model, criterion, ce_criterion, optimizer, args):
    losses = AverageMeter('Loss', ':.4e')
    model.train()

    for i, ((data, class_l, index_dic), d_idx) in enumerate(tqdm(train_loader)):
        concate_data = torch.cat(data, 0)
        concate_label = torch.cat([class_l] * 2 * args.baug).long()

        concate_label = concate_label.cuda()
        concate_data = concate_data.cuda()
        index_dic = torch.cat(index_dic, 0)


        if args.ESAugS:
            if args.classes == 7:
                sty_idx = [i for i, x in enumerate(index_dic) if x]
                if len(sty_idx) != 0:
                    for idx in sty_idx:
                        if concate_label[idx] == 0:
                                real_label = 'dog'
                        elif concate_label[idx] == 1:
                            real_label = 'elephant'
                        elif concate_label[idx] == 2:
                            real_label = 'giraffe'
                        elif concate_label[idx] == 3:
                            real_label = 'guitar'
                        elif concate_label[idx] == 4:
                            real_label = 'horse'
                        elif concate_label[idx] == 5:
                            real_label = 'house'
                        elif concate_label[idx] == 6:
                            real_label = 'person'

                        domain_name = random.sample(source_domain, 1)[0]
                        label_path = os.path.join(style_folder, domain_name, real_label)

                        concate_data[idx] = stylize_image(vgg, decoder, concate_data[idx], 10, label_path, 224)
            elif args.classes == 65:
                sty_idx = [i for i, x in enumerate(index_dic) if x]
                for idx in sty_idx:
                    read_label = main_path+ '/Art'
                    label_arr = os.listdir(read_label)
                    real_label = label_arr[int(concate_label[idx])]
                    domain_name = random.sample(source_domain, 1)[0]
                    label_path = os.path.join(main_path, domain_name, real_label)
                    concate_data[idx] = stylize_image(vgg, decoder, concate_data[idx], 10, label_path, 224)



        class_logit, features, _ = model(concate_data)

        if args.baug == 1:
            output_f, output_k = features[:args.batch_size, :], features[args.batch_size:, :]
            features = torch.stack((output_f, output_k), 1)
            sup_loss = criterion(features, concate_label[:args.batch_size * args.baug])
            ce_loss = ce_criterion(class_logit[:args.batch_size * args.baug],
                                   concate_label[:args.batch_size * args.baug])
            loss = ce_loss + sup_loss
        else:
            features = torch.stack(list(torch.reshape(features, (-1, args.batch_size * 2, 128))), 1)
            ce_loss = ce_criterion(class_logit[: args.batch_size * args.baug],
                                   concate_label[:args.batch_size * args.baug])
            sup_loss = criterion(features, concate_label[:args.batch_size * 2])
            loss = sup_loss + ce_loss

        # measure accuracy and record loss
        losses.update(loss.item(), int(concate_data.size(0) / 2))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.empty_cache()

    return losses.avg


def test_one_model(model, data_loader):
    model.eval()

    total_correct, total_num = 0, 0
    for batch_idx, ((data, class_l, name), _) in enumerate(tqdm(data_loader)):
        data, class_l = data.cuda(), class_l.cuda()
        with torch.no_grad():
            class_logit, _, _ = model(data)
            class_logit = F.softmax(class_logit, dim=1)

            pred_label = class_logit
            pred = pred_label.data.max(1, keepdim=True)[1].squeeze(1)
            correct = (pred == class_l).long()
            total_correct += correct.sum()
            total_num += correct.shape[0]

    accs = total_correct / float(total_num)

    return accs


def testmodel(model_1, model_2, model_3, data_loader):
    model_1.eval()
    model_2.eval()
    model_3.eval()

    total_correct, total_num = 0, 0

    for batch_idx, ((data, class_l, name), _) in enumerate(tqdm(data_loader)):
        data, class_l = data.cuda(), class_l.cuda()
        with torch.no_grad():
            class_logit_1, _, _ = model_1(data)
            class_logit_2, _, _ = model_2(data)
            class_logit_3, _, _ = model_3(data)

            class_logit_1 = F.softmax(class_logit_1, dim=1)
            class_logit_2 = F.softmax(class_logit_2, dim=1)
            class_logit_3 = F.softmax(class_logit_3, dim=1)

            pred_label = (class_logit_1 + class_logit_2 + class_logit_3) / 3
            pred = pred_label.data.max(1, keepdim=True)[1].squeeze(1)
            correct = (pred == class_l).long()
            total_correct += correct.sum()
            total_num += correct.shape[0]

    accs = total_correct / float(total_num)

    return accs



if __name__ == '__main__':
    main()
