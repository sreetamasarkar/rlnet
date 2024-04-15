"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample
from dataset.tiny_imagenet import get_tiny_imagenet_dataloaders, tiny_imagenet_c_testloader
from helper.util import adjust_learning_rate
from meters import  ScalarMeter, AverageMeter, flush_scalar_meters

from helper.loops import validate as validate_normal, validate_dualBN
from helper.loops_robust import validate_3BN
from helper.pretrain import init
from helper.attack import PGD, FGSM

from dataset.cifar10_deepaug import cifar_c_testloader
import numpy as np
from einops import reduce, repeat
import copy
from helper.util import accuracy
import torch.nn.functional as F
# from autoattack.autopgd_base import APGDAttack
'''
Reference: https://github.com/VITA-Group/AugMax/blob/176acdb0624060b64acbeed130e435303f14a63c/test.py
'''
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

## Test on Tiny-ImageNet:
ResNet18_c_CE_list = [
    0.8037, 0.7597, 0.7758, 0.8426, 0.8274, 
    0.7907, 0.8212, 0.7497, 0.7381, 0.7433, 
    0.6800, 0.8939, 0.7308, 0.6121, 0.6452
]

def find_mCE(target_model_c_CE, anchor_model_c_CE):
    '''
    Args:
        target_model_c_CE: np.ndarray. shape=(15). CE of each corruption type of the target model.
        anchor_model_c_CE: np.ndarray. shape=(15). CE of each corruption type of the anchor model (normally trained ResNet18 as default).
    '''
    assert len(target_model_c_CE) == 15 # a total of 15 types of corruptions
    mCE = 0
    for target_model_CE, anchor_model_CE in zip(target_model_c_CE, anchor_model_c_CE):
        mCE += target_model_CE/anchor_model_CE
    mCE /= len(target_model_c_CE)
    return mCE

def val_tin_c(args, model, criterion_cls, n_cls=10, mask_list = None, use2BN=False, use3BN=False, val_lambda=None):
    '''
    Evaluate on Tiny ImageNet-C
    '''
    fp = open(args.outfile, 'a+')
    if mask_list is not None:
        mask_list_copy = copy.deepcopy(mask_list)

    test_seen_c_loader_list = []
    for corruption in CORRUPTIONS:
        test_seen_c_loader_list_c = []
        for severity in range(1,6):
            test_c_loader_c_s = tiny_imagenet_c_testloader(data_dir=os.path.join('/data1', 'Tiny-ImageNet-C/'),
                corruption=corruption, severity=severity, 
                test_batch_size=args.batch_size, num_workers=args.num_workers)
            test_seen_c_loader_list_c.append(test_c_loader_c_s)
        test_seen_c_loader_list.append(test_seen_c_loader_list_c)

    model.eval()
    # val corruption:
    print('evaluating corruptions...')
    test_CE_c_list = []
    for corruption, test_seen_c_loader_list_c in zip(CORRUPTIONS, test_seen_c_loader_list):
        test_c_CE_c_s_list = []
        ts = time.time()
        for severity in range(1,6):
            test_c_loader_c_s = test_seen_c_loader_list_c[severity-1]
            test_c_batch_num = len(test_c_loader_c_s)
            # print(test_c_batch_num) # each corruption has 10k * 5 images, each magnitude has 10k images
            test_c_loss_meter, test_c_CE_meter = AverageMeter(), AverageMeter()
            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(test_c_loader_c_s):
                    images, targets = images.cuda(), targets.cuda()
                    features = [] 
                    idx2BN = val_lambda
                    if mask_list == None:
                        logits, _ = model(images, features, is_feat=False, idx2BN=idx2BN)
                    else:
                        for mask_index, mask in enumerate(mask_list):
                            mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(images.shape)[0]) # b c h w
                        logits, _ = model(images, mask_list, features, is_feat=False, idx2BN=idx2BN)
                        mask_list = copy.deepcopy(mask_list_copy)
                    # logits = model(images)
                    target = targets[targets < n_cls]
                    logits = logits[targets < n_cls]
                    loss = F.cross_entropy(logits, target)
                    pred = logits.data.max(1)[1]
                    ce = (~pred.eq(target.data)).float().mean()
                    # append loss:
                    test_c_loss_meter.append(loss.item())
                    test_c_CE_meter.append(ce.item())
            
            # test loss and acc of each type of corruptions:
            test_c_CE_c_s = test_c_CE_meter.avg
            test_c_CE_c_s_list.append(test_c_CE_c_s)
        test_CE_c = np.mean(test_c_CE_c_s_list)
        test_CE_c_list.append(test_CE_c)

        # print
        print('%s test time: %.2fs' % (corruption, time.time()-ts))
        corruption_str = '%s CE: %.4f' % (corruption, test_CE_c)
        print(corruption_str)
        fp.write(corruption_str + '\n')
        fp.flush()
    # mean over 16 types of corruptions:
    test_c_acc = 1-np.mean(test_CE_c_list)
    # weighted mean over 16 types of corruptions:
    test_mCE = find_mCE(test_CE_c_list, anchor_model_c_CE=ResNet18_c_CE_list)

    # print
    avg_str = 'corruption acc: %.4f' % (test_c_acc)
    print(avg_str)
    fp.write(avg_str + '\n')
    mCE_str = 'mCE: %.4f' % test_mCE
    print(mCE_str)
    fp.write(mCE_str + '\n')
    fp.flush()


def val_cifar_c(args, model, criterion_cls, n_cls=10, mask_list = None, use2BN=False, use3BN=False, val_lambda=None):
    '''
    Evaluate on CIFAR10/100-C
    '''
    fp = open(args.outfile, 'a+')
    if mask_list is not None:
        mask_list_copy = copy.deepcopy(mask_list)

    test_seen_c_loader_list = []
    for corruption in CORRUPTIONS:
        test_c_loader, n_data = cifar_c_testloader(corruption=corruption, data_dir='./data/', num_classes=n_cls, 
            test_batch_size=args.batch_size, num_workers=args.num_workers)
        test_seen_c_loader_list.append(test_c_loader)

    model.eval()
    # val corruption:
    print('evaluating corruptions...')
    test_c_losses, test_c_accs = [], []
    for corruption, test_c_loader in zip(CORRUPTIONS, test_seen_c_loader_list):
        test_c_batch_num = len(test_c_loader)
        # print(test_c_batch_num) # each corruption has 10k * 5 images, each magnitude has 10k images
        ts = time.time()
        test_c_loss_meter, test_c_acc_meter = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_c_loader):
                images, targets = images.cuda(), targets.cuda()
                features = []
                idx2BN = None
                # if use2BN:
                #     idx2BN = int(targets.size()[0]) if val_lambda==0 else 0   
                # elif use3BN:
                idx2BN = val_lambda
                if mask_list == None:
                    logits, _ = model(images, features, is_feat=False, idx2BN=idx2BN)
                else:
                    for mask_index, mask in enumerate(mask_list):
                        mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(images.shape)[0]) # b c h w
                    logits, _ = model(images, mask_list, features, is_feat=False, idx2BN=idx2BN)
                    mask_list = copy.deepcopy(mask_list_copy)
                # logits = model(images)
                loss = criterion_cls(logits, targets)
                pred = logits.data.max(1)[1]
                acc = pred.eq(targets.data).float().mean()
                # acc, acc5 = accuracy(logits, targets, topk=(1, 5))
                
                # append loss:
                test_c_loss_meter.append(loss.item())
                test_c_acc_meter.append(acc.item())

        print('%s test time: %.2fs' % (corruption, time.time()-ts))
        # test loss and acc of each type of corruptions:
        test_c_losses.append(test_c_loss_meter.avg)
        test_c_accs.append(test_c_acc_meter.avg)

        # print
        corruption_str = '%s: %.4f' % (corruption, test_c_accs[-1])
        print(corruption_str)
        fp.write(corruption_str + '\n')
        fp.flush()
    # mean over 16 types of attacks:
    test_c_loss = np.mean(test_c_losses)
    test_c_acc = np.mean(test_c_accs)

    # print
    avg_str = 'corruption acc: (mean) %.4f' % (test_c_acc)
    print(avg_str)
    fp.write(avg_str + '\n')
    fp.flush()


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar100', 'cifar10', 'tiny_imagenet'], help='dataset')


    # model
    # The  resnet50,38,26 matches the models of SASA paper and
    # are written by us. The ResNet50 model is of original CRD paper
    parser.add_argument('--model', type=str, default='resnet8',
                        choices=['wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'wrn_22_8',
                                 'ODResNet18', 'ODResNet34', 
                                 'ResNet50', 'ResNet18', 'ResNet18_3BN', 'ResNet34', 'ResNet34_3BN',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'Custom_wrn_22_8',
                                 'CustomResNet18', 'CustomResNet18_3BN', 'CustomResNet34', 'CustomODvgg16', 'Custom_wrn_22_8_3BN'])
    parser.add_argument('--path', type=str, default=None, help='teacher model snapshot')

    parser.add_argument('--validate_mode', type=str, default=None, choices=['Normal_Network', 'Custom_Network'])

    # attack parameters
    parser.add_argument('--attack_mode', type=str, default=None, choices=['pgd', 'fgsm'])
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--steps', type=int, default=7)
    parser.add_argument('--use2BN', action='store_true', help='If true, use dual BN')
    parser.add_argument('--use3BN', action='store_true', help='If true, use triple BN')

    opt = parser.parse_args()

    opt.save_folder_path = './save/common_corruptions/'

    # opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    # opt.save_folder = os.path.join(opt.save_folder_path, opt.path.split('/')[-2])
    opt.save_folder = os.path.dirname(opt.path)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # save terminal output to a file
    opt.outfile = os.path.join(opt.save_folder,'common_corruption_results.txt')

    return opt



def load_model(model_path, n_cls, opt):
    print('==> loading model')
    print(opt.model)
    model_name = opt.model
    if opt.use2BN:
        model = model_dict[opt.model](num_classes=n_cls, use2BN=opt.use2BN)
    elif opt.use3BN:
        model = model_dict[opt.model](num_classes=n_cls, use3BN=opt.use3BN)
    else:
        model = model_dict[opt.model](num_classes=n_cls)    
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    if opt.validate_mode == 'Normal_Network' or opt.validate_mode == 'OD_Network':
        return model
    elif opt.validate_mode == 'Custom_OD_Network' or opt.validate_mode == 'Custom_Network':
        return model, torch.load(model_path)['mask_epoch']

def cal_mask_relu(mask_list):
    total_relu = 0
    OD_total_relu = 0

    for i in range(len(mask_list)):
        total_relu += mask_list[i].sum()
        mask_fore, mask_behind = mask_list[i].split(int(mask_list[i].shape[0]/2), 0)
        OD_total_relu += mask_fore.sum()
    print('====================================================================')
    print('Total_relu:', float(total_relu))
    print('The first half channel relu:', float(OD_total_relu))
    print('====================================================================')

def cal_nonmask_relu(model, idx2BN=None):
    data = torch.randn(2, 3, 32, 32).cuda()
    features = []
    model.eval()
    with torch.no_grad():
        out_t, features = model(data, features, is_feat = False, idx2BN=idx2BN)
    
    counts = 0
    for feature_idx in range(len(features)):
        counts = counts + features[feature_idx].shape[1] * features[feature_idx].shape[2] * features[feature_idx].shape[3]
    print('==========================================')
    print('Total counts of relus:', counts)
            

def main():

    # torch.cuda.set_device(2)
    opt = parse_option()

    attacker = None
    # criterion_div = DistillKL(1)
    if opt.attack_mode == 'pgd': 
        attacker = PGD(eps=opt.eps, steps=opt.steps)
        # attacker = PGD(eps=opt.eps, steps=opt.steps, criterion=criterion_div)

    elif opt.attack_mode == 'fgsm': attacker = FGSM(eps=opt.eps)
    
    if opt.validate_mode == 'Custom_OD_Network':
        opt.od_training = True

    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True, attacker=attacker)
        n_cls = 100
    elif opt.dataset == 'cifar10':

        train_loader, val_loader, n_data = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True, attacker=attacker)
        n_cls = 10
    elif opt.dataset == 'tiny_imagenet':

        train_loader, val_loader, n_data = get_tiny_imagenet_dataloaders(batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True, attacker=attacker)
        n_cls = 200
    # model
    # import pdb; pdb.set_trace()
    mask_epoch = None
    if opt.validate_mode == 'Normal_Network' or opt.validate_mode == 'OD_Network':
        model = load_model(opt.path, n_cls, opt)
    elif opt.validate_mode == 'Custom_OD_Network' or opt.validate_mode == 'Custom_Network':
        model, mask_epoch = load_model(opt.path, n_cls, opt)
        
    # model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)


    criterion_cls = nn.CrossEntropyLoss()

    
    if torch.cuda.is_available():
        model.cuda()
        criterion_cls.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    if opt.validate_mode == 'Normal_Network':
        if opt.use2BN:
            relu_count  = cal_nonmask_relu(model, idx2BN=0.0)
            print('relu count:',relu_count)
            test_acc, robust_test_acc, _ = validate_dualBN(val_loader, model, criterion_cls, opt, attacker=attacker)
            print('natural accuracy (lambda = 0.0): ', test_acc[0.0].avg)
            print('natural accuracy (lambda = 1.0): ', test_acc[1.0].avg)
            print('robust accuracy (lambda = 0.0): ', robust_test_acc[0.0].avg)
            print('robust accuracy (lambda = 1.0): ', robust_test_acc[1.0].avg)
        elif opt.use3BN:
            test_acc, robust_test_acc, _ = validate_3BN(val_loader, model, criterion_cls, opt, attacker=attacker)
            print('natural accuracy (lambda = 0.0): ', test_acc[0.0].avg)
            print('natural accuracy (lambda = 2.0): ', test_acc[2.0].avg)
            print('robust accuracy (lambda = 0.0): ', robust_test_acc[0.0].avg)
            print('robust accuracy (lambda = 2.0): ', robust_test_acc[2.0].avg)
        else:
            relu_count  = cal_nonmask_relu(model)
            print('relu count:',relu_count)
            teacher_acc, robust_acc, _, _ = validate_normal(val_loader, model, criterion_cls, opt, attacker=attacker)
            print('==========================================================')
            print('The acc of {} is:{}'.format(opt.model, str(float(teacher_acc))))
            if attacker:
                print('The robust acc of {} is:{}'.format(opt.model, str(float(robust_acc))))

    elif opt.validate_mode == 'Custom_Network':
        cal_mask_relu(mask_epoch)
        if opt.use2BN:
            test_acc, robust_test_acc, _ = validate_dualBN(val_loader, model, criterion_cls, opt, mask_epoch, attacker=attacker)
            print('natural accuracy (lambda = 0.0): ', test_acc[0.0].avg)
            print('natural accuracy (lambda = 1.0): ', test_acc[1.0].avg)
            print('robust accuracy (lambda = 0.0): ', robust_test_acc[0.0].avg)
            print('robust accuracy (lambda = 1.0): ', robust_test_acc[1.0].avg)
        elif opt.use3BN:
            test_acc, robust_test_acc, _ = validate_3BN(val_loader, model, criterion_cls, opt, mask_epoch, attacker=attacker)
            print('natural accuracy (lambda = 0.0): ', test_acc[0.0].avg)
            print('natural accuracy (lambda = 2.0): ', test_acc[2.0].avg)
            print('robust accuracy (lambda = 0.0): ', robust_test_acc[0.0].avg)
            print('robust accuracy (lambda = 2.0): ', robust_test_acc[2.0].avg)
        else:
            teacher_acc, robust_acc, _, _ = validate_normal(val_loader, model, criterion_cls, opt, mask_epoch, attacker)
            print('==========================================================')
            print('The natural acc of {} is:{}'.format(opt.model, str(float(teacher_acc))))
            if attacker:
                print('The robust acc of {} is:{}'.format(opt.model, str(float(robust_acc))))

    # validate auto-attack
    # from autoattack.autoattack import AutoAttack
    # adversary = AutoAttack(model, norm='Linf', eps=8./255., log_path=opt.outfile,
    #     version='standard')
    # # run attack 
    # l = [x for (x, y) in val_loader]
    # x_test = torch.cat(l, 0)
    # l = [y for (x, y) in val_loader]
    # y_test = torch.cat(l, 0)
    
    # with torch.no_grad():
    #     adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=opt.batch_size)

    if opt.dataset == 'tiny_imagenet':
        if opt.use2BN:
            val_tin_c(opt, model, criterion_cls, mask_list=mask_epoch, n_cls=n_cls, use2BN=True, val_lambda=0.0) 
        else:
            val_tin_c(opt, model, criterion_cls, mask_list=mask_epoch, n_cls=n_cls) 
    else:
        if opt.use2BN:
            val_cifar_c(opt, model, criterion_cls, mask_list=mask_epoch, n_cls=n_cls, use2BN=True, val_lambda=0.0) 
        elif opt.use3BN:
            val_cifar_c(opt, model, criterion_cls, mask_list=mask_epoch, n_cls=n_cls, use3BN=True, val_lambda=1.0) 
        else:
            val_cifar_c(opt, model, criterion_cls, mask_list=mask_epoch, n_cls=n_cls)    



if __name__ == '__main__':
    main()


