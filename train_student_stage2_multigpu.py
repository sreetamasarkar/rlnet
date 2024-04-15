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

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample, get_cifar100_dataloaders_augmix
from dataset.cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample, get_cifar10_dataloaders_augmix
from dataset.imagenet import get_imagenet_dataloader, get_dataloader_sample
from dataset.tiny_imagenet import get_tiny_imagenet_dataloaders, get_tiny_imagenet_dataloaders_augmix
from helper.util import adjust_learning_rate

#SK:added the GRAMM loss to the distill zoo and imported here
from distiller_zoo import DistillKL, Attention

# from helper.loops import train_distill_stage2 as train, validate
from helper.loops import train_distill_stage2 as train, validate, validate_dualBN
from helper.loops_robust import train_distill_stage2_3BN, train_distill_stage2_2BN, validate_3BN
from helper.attack import PGD, FGSM
from helper.pretrain import init
from helper.loops_trainable_mask import train_distill_stage2 as train_distill_stage2_TM
from admm_core import Masking, CosineDecay, LinearDecay, add_sparse_args

from einops import rearrange, reduce, repeat
import yaml
import sys

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--t2_epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--t1_epochs', type=int, default=240, help='number of first teacher epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'imagenet', 'tiny_imagenet'], help='dataset')

    # model
    # The  resnet50,38,26 matches the models of SASA paper and
    # are written by us. The ResNet50 model is of original CRD paper
    parser.add_argument('--model_s', type=str, default='CustomResNet18',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'resnet50', 'resnet38', 'resnet26',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'CustomResNet18_3BN', 'CustomResNet18_TM',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'CustomResNet18', 'CustomResNet34', 'Custom_wrn_22_8', 'Custom_wrn_22_8_3BN'])
    parser.add_argument('--path_t', type=str, default='ResNet18', help='teacher model snapshot')
    parser.add_argument('--path_s_load', type=str, default='./save_folder/CustomResNet18_stage1_best.pth', help='student model snapshot')
    parser.add_argument('--student_setting_id', type=int, default=0, help='student model snapshot')
    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention',
                                                                    'similarity','correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=0.5, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0.5, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=1000, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])
    
    #self attention related
    parser.add_argument('--self_attn', type=bool, default=False, help='choose when you want to select an attn student model')
    parser.add_argument('--stem', type=bool, default=False, help='choose the stem to be attention model as well or not')

    parser.add_argument('--decay_schedule', type=str, default='linear')

    # attack parameters
    parser.add_argument('--attack_mode', type=str, default=None, choices=['pgd', 'fgsm'])
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--steps', type=int, default=7)
    
    parser.add_argument("--seed", help="set random generator seed", default='0')

    parser.add_argument('--robust_train_mode', type=str, default=None, choices=['trades', 'rslad', 'pgd', 'rslad+attn', 'kl', 'ard', 'rslad+attn+augattn', 
                                                                                'rslad+attn+augattn+ce', 'rslad+attn+augattn+advattn', 'rslad+attn+augattn+advattn+ce'])
    parser.add_argument('--augment_mode', type=str, default=None, choices=['augmix', 'deepaug'])
    parser.add_argument('--use2BN', action='store_true', help='If true, use dual BN')
    parser.add_argument('--use3BN', action='store_true', help='If true, use triple BN')
    parser.add_argument('--trainable_mask', action='store_true', help='If true, use trainable mask')
    parser.add_argument('--sensitivity', type=str, default='ResNet18_C10_relu82k_sensitivity')

    parser.add_argument("--local-rank", default=0, type=int)
    
    add_sparse_args(parser)
    opt = parser.parse_args()
    torch.manual_seed(int(opt.seed))
    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model/stage2'
        opt.tb_path = '/path/to/my/student_tensorboards/stage2'
    else:
        opt.model_path = './save/student_model/stage2'
        opt.tb_path = './save/student_tensorboards/stage2'

    if opt.trainable_mask:
        opt.model_path += '/trainable_mask'
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)
    model_tname = opt.model_t
    if '3BN' in opt.path_t:
        model_tname = opt.model_t + '3BN'
    # if 'pgd' in opt.path_t:
    #     model_tname = opt.model_t + 'pgd'
    if '3BN' in opt.path_s_load.split('/')[-2].split('_')[-1]:
        stage1 = '3BN'
    elif '2BN' in opt.path_s_load.split('/')[-2].split('_')[-1]:
        stage1 = '2BN'
    else:
        stage1 = 'senet'
    # opt.model_name = 'S:{}_T1:{}_{}_{}_lr:{}_r:{}_a:{}_b:{}_{}_stem:{}_dense:{}_stage1:{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
    #                                                     opt.learning_rate, opt.gamma, opt.alpha, opt.beta, opt.trial, opt.stem, opt.dense, opt.student_setting_id)
    relu_count = opt.path_s_load.split('k')[0].split('_')[-1] + 'k'
    # if opt.attack_mode: opt.model_name += opt.attack_mode
    # opt.model_name = 'S:{}_T1:{}_{}_lr:{}_a:{}_{}_distill:{}_kdT:{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.learning_rate, opt.alpha, relu_count, opt.distill, opt.kd_T)
    opt.model_name = 'S:{}_T1:{}_stage1:{}_{}_{}'.format(opt.model_s, model_tname, stage1, opt.dataset, relu_count)
    # opt.model_name = 'S:{}_T1:{}_stage1:{}_{}_{}_loss:ce_augmix'.format(opt.model_s, model_tname, stage1, opt.dataset, relu_count)
    # opt.model_name = 'S:{}_T1:{}_stage1:{}_{}_{}_loss:ce_kl_augmix'.format(opt.model_s, model_tname, stage1, opt.dataset, relu_count)
    # opt.model_name = 'S:{}_T1:{}_stage1:{}_{}_{}_loss:ce_kl_attn_augmix'.format(opt.model_s, model_tname, stage1, opt.dataset, relu_count)
    if opt.augment_mode:
        opt.model_name += '_' + opt.augment_mode
    if opt.robust_train_mode:
        opt.model_name += '_' + opt.robust_train_mode
        # opt.model_name += '+augattn'
    else:
        opt.model_name += '_senet' 
    opt.model_name += '_kdT{}'.format(opt.kd_T) 
    if opt.use3BN:
        opt.model_name += '_3BN'
    elif opt.use2BN:
        opt.model_name += '_2BN'
    opt.model_name += '_batch{}'.format(opt.batch_size)
    # opt.model_name += '_attn+augattn+advattn'
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    # save terminal output to a file
    opt.outfile = os.path.join(opt.save_folder,'results.out')
    return opt

def get_teacher_name(model_path):
    """parse teacher name"""
    #SK:changed the [-2] to [2] for corresponding model name fetch from saved location in our case
    segments = model_path.split('/')[-1].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

def get_student_name(model_path):
    """parse student name"""
    return model_path.split('/')[3].split('_')[0][2:]

def load_teacher(opt, model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    if opt.use3BN:
        model_t += '_3BN'
        model = model_dict[model_t](num_classes=n_cls, use3BN=opt.use3BN)
    elif opt.use2BN:
        model = model_dict[model_t](num_classes=n_cls, use2BN=opt.use2BN)
    else:
        model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    #model.load_state_dict(torch.load(model_path))
    # pretrained_model_state_dict =timm.create_model(model_t.lower(), num_classes=n_cls, pretrained=True).state_dict()
    # for key, val in pretrained_model_state_dict.copy().items():
    #     newkey = None
    #     if 'fc' in key:
    #         newkey = key.replace('fc', 'linear')

    #     if 'downsample' in key:
    #         newkey = key.replace('downsample', 'shortcut')

    #     if newkey:
    #         pretrained_model_state_dict[newkey] = val

    #         del pretrained_model_state_dict[key]

    # # conv1 in resnet18 has different size
    # model_dic = model.state_dict()
    # # pretrained_dict = {k: v for k, v in pretrained_model_state_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
    # model_dic.update(pretrained_model_state_dict)

    # model.load_state_dict(model_dic)
    # #print (model)
    print('==> done')
    return model

def load_student(opt, model_path, teacher_path, n_cls):
    print('==> loading student model')
    model_s = opt.model_s
    if opt.use3BN:
        model = model_dict[model_s](num_classes=n_cls, use3BN=opt.use3BN)
    elif opt.use2BN:
        model = model_dict[model_s](num_classes=n_cls, use2BN=opt.use2BN)
    else:
        model = model_dict[model_s](num_classes=n_cls)
    # model.load_state_dict(torch.load(teacher_path)['model'])
    model.load_state_dict(torch.load(model_path)['model'])

    # Replace BN layers
    # pretrained_model = torch.load(model_path)['model']
    # for key, val in pretrained_model.copy().items():
    #     if 'bn1' in key:
    #         bn1key = key.replace('bn1', 'bn1.BN_c')
    #         pretrained_model[bn1key] = val
    #         del pretrained_model[key]
    #     if 'bn2' in key:     
    #         bn2key = key.replace('bn2', 'bn2.BN_c')
    #         pretrained_model[bn2key] = val
    #         del pretrained_model[key]
    # model_dic = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dic}
    # model_dic.update(pretrained_dict)
    # model.load_state_dict(model_dic)
    print('==> done')
    return model, torch.load(model_path)['mask_epoch']

def set_param_grad_off(module):
    for param_name, param in module.named_parameters():
        if ('scores' in param_name):
            param.requires_grad = False

def main():
    best_acc = 0
    best_robust_acc = 0
    opt = parse_option()
    # f = open(opt.outfile, 'w')
    # sys.stdout = f
    print(opt)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    
    attacker = None
    attacker_val = None
    if opt.attack_mode == 'pgd':
        if 'rslad' in opt.robust_train_mode or 'trades' in opt.robust_train_mode:
            attacker = PGD(eps=opt.eps, steps=opt.steps, criterion=criterion_div)
        else:
            attacker = PGD(eps=opt.eps, steps=opt.steps)
        # Validate using PGD-20 
        attacker_val = PGD(eps=opt.eps, steps=opt.steps)    
    elif opt.attack_mode == 'fgsm': attacker = FGSM(eps=opt.eps)

    # tensorboard logger
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # distributed settings
    opt.distributed = False
    if 'WORLD_SIZE' in os.environ:
        opt.distributed = int(os.environ['WORLD_SIZE']) > 1
        print (f"int(os.environ['WORLD_SIZE']):{int(os.environ['WORLD_SIZE'])}")
    opt.device = 'cuda:0'
    opt.world_size = 1
    opt.rank = 0  # global rank
    print ('Distributed training: ',opt.distributed)
    if opt.distributed:
        opt.device = 'cuda:%d' % opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        opt.world_size = torch.distributed.get_world_size()
        opt.rank = torch.distributed.get_rank()
        #logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
        #             % (opt.rank, opt.world_size))
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d %d, total %d.'
                     % (opt.rank, opt.local_rank, opt.world_size))
    else:
        #logger.info('Training with a single process on 1 GPUs.')
        print ('Training with a single process on 1 GPUs.')
    assert opt.rank >= 0

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.augment_mode == 'augmix':
            train_loader, val_loader, n_data = get_cifar100_dataloaders_augmix(batch_size=opt.batch_size, num_workers=opt.num_workers, attacker=attacker, distributed=opt.distributed, is_instance=True)
        else:    
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True,
                                                                        attacker = attacker)
        n_cls = 100
    elif opt.dataset == 'cifar10':
        if opt.augment_mode == 'augmix':
            train_loader, val_loader, n_data = get_cifar10_dataloaders_augmix(batch_size=opt.batch_size, num_workers=opt.num_workers, distributed=opt.distributed, is_instance=True, attacker=attacker)
        else:    
            train_loader, val_loader, n_data = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        distributed=opt.distributed,
                                                                        is_instance=True,
                                                                        attacker=attacker)
        n_cls = 10
    elif opt.dataset == 'tiny_imagenet':
        if opt.augment_mode == 'augmix':
            train_loader, val_loader, n_data = get_tiny_imagenet_dataloaders_augmix(batch_size=opt.batch_size, num_workers=opt.num_workers, attacker=attacker, distributed=opt.distributed, is_instance=True)
        else:
            train_loader, val_loader, n_data = get_tiny_imagenet_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, attacker=attacker, distributed=opt.distributed, is_instance=True)
        n_cls=200
    elif opt.dataset == 'imagenet':
        train_loader, val_loader, n_data = get_imagenet_dataloader(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        distributed=opt.distributed,
                                                                        is_instance=True)
        n_cls=1000
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt, opt.path_t, n_cls)

    if opt.trainable_mask:
        s = open('sensitivity_list.yaml', 'r')
        stream = s.read()
        sensitivity_yaml = yaml.safe_load(stream)
        opt.sensitivity_list = sensitivity_yaml[opt.sensitivity]  
        model_s = model_dict[opt.model_s](num_classes=n_cls, sensitivity_list=opt.sensitivity_list)
    else:
        if opt.use3BN:
            model_s = model_dict[opt.model_s](num_classes=n_cls, use3BN=opt.use3BN)
        elif opt.use2BN:
            model_s = model_dict[opt.model_s](num_classes=n_cls, use2BN=opt.use2BN)
        else:
            model_s = model_dict[opt.model_s](num_classes=n_cls)
   
        model_s, mask_list = load_student(opt, opt.path_s_load, opt.path_t, n_cls)
    
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay

    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True
        if not opt.trainable_mask:
            mask_list = [cur_mask.cuda() for cur_mask in mask_list]

    # setup distributed training
    if opt.distributed:
        if opt.local_rank == 0:
            print("Using native Torch DistributedDataParallel.")
        for module_idx, module in enumerate(module_list):
            module_list[module_idx] = torch.nn.parallel.DistributedDataParallel(module_list[module_idx], device_ids=[opt.local_rank])

        #for index, mask in enumerate(mask_list):
        #    mask_list[index] = torch.nn.parallel.DistributedDataParallel(mask_list[index], device_ids=[opt.local_rank])

    # validate teacher accuracy
    if opt.use3BN:
        test_acc, robust_test_acc, _ = validate_3BN(val_loader, model_t, criterion_cls, opt, attacker=attacker_val)
        print('natural teacher accuracy (lambda = 0.0): ', test_acc[0.0].avg)
        print('natural teacher accuracy (lambda = 2.0): ', test_acc[2.0].avg)
        print('robust teacher accuracy (lambda = 0.0): ', robust_test_acc[0.0].avg)
        print('robust teacher accuracy (lambda = 2.0): ', robust_test_acc[2.0].avg)
    elif opt.use2BN:
        test_acc, robust_test_acc, _ = validate_dualBN(val_loader, model_t, criterion_cls, opt, attacker=attacker_val)
        print('natural teacher accuracy (lambda = 0.0): ', test_acc[0.0].avg)
        print('natural teacher accuracy (lambda = 1.0): ', test_acc[1.0].avg)
        print('robust teacher accuracy (lambda = 0.0): ', robust_test_acc[0.0].avg)
        print('robust teacher accuracy (lambda = 1.0): ', robust_test_acc[1.0].avg)
    else:
        teacher_acc, teacher_acc_robust, _, _ = validate(val_loader, model_t, criterion_cls, opt, attacker=attacker_val)
        print('teacher accuracy: ', teacher_acc)
        if attacker:
            print('robust teacher accuracy: ', teacher_acc_robust)

    if not opt.trainable_mask:
        if opt.use3BN:
            test_acc, robust_test_acc, _ = validate_3BN(val_loader, model_s, criterion_cls, opt, mask_list, attacker=attacker_val)
            print('natural stage1 accuracy (lambda = 0.0): ', test_acc[0.0].avg)
            print('natural stage1 accuracy (lambda = 2.0): ', test_acc[2.0].avg)
            print('robust stage1 accuracy (lambda = 0.0): ', robust_test_acc[0.0].avg)
            print('robust stage1 accuracy (lambda = 2.0): ', robust_test_acc[2.0].avg)
        elif opt.use2BN:
            test_acc, robust_test_acc, _ = validate_dualBN(val_loader, model_s, criterion_cls, opt, mask_list, attacker=attacker_val)
            print('natural stage1 accuracy (lambda = 0.0): ', test_acc[0.0].avg)
            print('natural stage1 accuracy (lambda = 1.0): ', test_acc[1.0].avg)
            print('robust stage1 accuracy (lambda = 0.0): ', robust_test_acc[0.0].avg)
            print('robust stage1 accuracy (lambda = 1.0): ', robust_test_acc[1.0].avg)
        else:
            teacher_acc, teacher_acc_robust, _, _ = validate(val_loader, model_s, criterion_cls, opt, mask_list, attacker=attacker_val)
            print('student stage 1 accuracy: ', teacher_acc)
            if attacker:
                print('robust student stage 1 accuracy: ', teacher_acc_robust)

    

    # routine
    mask_weight = None
    if not opt.dense:
        if opt.decay_schedule == 'cosine':
            decay = CosineDecay(opt.prune_rate, len(train_loader)*(opt.t2_epochs))
        elif opt.decay_schedule == 'linear':
            decay = LinearDecay(opt.prune_rate, len(train_loader)*(opt.t2_epochs))
        print('using {} decay schedule'.format(opt.decay_schedule))

        mask_weight = Masking(optimizer, decay, prune_rate=opt.prune_rate, prune_mode=opt.prune, \
            growth_mode=opt.growth, redistribution_mode=opt.redistribution, verbose=opt.verbose)
        mask_weight.add_module(model_s, density=opt.density)

    if opt.use3BN or opt.use2BN:
        if opt.use3BN:
            val_lambdas = [0.0, 2.0]
        else:
            val_lambdas = [0.0, 1.0]
        best_TA, best_ATA = {}, {}
        for val_lambda in val_lambdas:
            best_TA[val_lambda], best_ATA[val_lambda] = 0, 0

    for epoch in range(1, opt.t2_epochs+1):
        if opt.trainable_mask:
            if epoch == 2: #at the end of first epoch
                model_s.load_state_dict(torch.load(opt.path_s_load)['model'])
                print('Loaded stage-1 model')
                teacher_acc, teacher_acc_robust, _, _ = validate(val_loader, model_s, criterion_cls, opt, attacker=attacker_val)
                print('student stage 1 accuracy: ', teacher_acc)
                if attacker:
                    print('robust student stage 1 accuracy: ', teacher_acc_robust)
                # set masks as non-trainable parameters
                set_param_grad_off(model_s)    
        if opt.distributed:
            train_loader.sampler.set_epoch(epoch)
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        if opt.trainable_mask:
            train_acc, robust_train_acc, train_loss, mask_epoch = train_distill_stage2_TM(epoch, train_loader, module_list, criterion_list, optimizer, opt, attacker=attacker)
        elif opt.use3BN:
            train_acc, robust_train_acc, train_loss, mask_epoch = train_distill_stage2_3BN(epoch, train_loader, module_list, criterion_list, optimizer, opt, mask_list, mask_weight, attacker=attacker)
        elif opt.use2BN:
            train_acc, robust_train_acc, train_loss, mask_epoch = train_distill_stage2_2BN(epoch, train_loader, module_list, criterion_list, optimizer, opt, mask_list, mask_weight, attacker=attacker)
        else:
            train_acc, robust_train_acc, train_loss, mask_epoch = train(epoch, train_loader, module_list, criterion_list, optimizer, opt, mask_list, mask_weight, attacker=attacker)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # if opt.local_rank == 0:
        #     logger.log_value('train_acc', train_acc, epoch)
        #     logger.log_value('train_loss', train_loss, epoch)
        #     if attacker:
        #         logger.log_value('robust_train_acc', robust_train_acc, epoch)

        time_val_start = time.time()
        if opt.use3BN:
            test_acc, robust_test_acc, mask_list = validate_3BN(val_loader, model_s, criterion_cls, opt, mask_epoch, attacker=attacker_val)
        elif opt.use2BN:
            test_acc, robust_test_acc, mask_list = validate_dualBN(val_loader, model_s, criterion_cls, opt, mask_epoch, attacker=attacker_val)
        else:
            test_acc, robust_test_acc, test_loss, mask_list = validate(val_loader, model_s, criterion_cls, opt, mask_epoch, attacker=attacker_val)

        if opt.local_rank == 0:
            # save the best model
            if opt.use3BN or opt.use2BN:
                val_str = 'Epoch %d | Validation | Time: %.4f\n' % (epoch, (time.time()-time_val_start))
                for val_lambda in val_lambdas:
                    val_str += 'val_lambda%s: TA: %.4f, ATA: %.4f\n' % (val_lambda, test_acc[val_lambda].avg, robust_test_acc[val_lambda].avg)
                    if test_acc[val_lambda].avg >= best_TA[val_lambda]:
                        best_TA[val_lambda] = test_acc[val_lambda].avg # update best TA
                        state = {
                            'epoch': epoch,
                            'model': model_s.state_dict(),
                            'best_acc': best_TA[val_lambda],
                            'robust_acc': robust_test_acc[val_lambda].avg, 
                            'mask_epoch': mask_list
                        }
                        save_file = os.path.join(opt.save_folder, '{}_stage2_best_TA{}.pth'.format(opt.model_s, val_lambda))
                        print('saving the best model!')
                        torch.save(state, save_file)
                    if robust_test_acc[val_lambda].avg >= best_ATA[val_lambda]:
                        best_ATA[val_lambda] = robust_test_acc[val_lambda].avg # update best ATA
                        state = {
                            'epoch': epoch,
                            'model': model_s.state_dict(),
                            'best_robust_acc': best_ATA[val_lambda],
                            'natural acc': test_acc[val_lambda].avg,
                            'mask_epoch': mask_list
                        }
                        save_file = os.path.join(opt.save_folder, '{}_stage2_best_ATA{}.pth'.format(opt.model_s, val_lambda))
                        print('saving the best robust model!')
                        torch.save(state, save_file)
                
                val_fp = open(os.path.join(opt.save_folder, 'val_log.txt'), 'a+')
                print(val_str)
                val_fp.write(val_str + '\n')
                val_fp.close()
            else:
                # logger.log_value('test_acc', test_acc, epoch)
                # logger.log_value('test_loss', test_loss, epoch)
                # if attacker:
                #     logger.log_value('robust_test_acc', robust_test_acc, epoch)

                if test_acc > best_acc:
                    best_acc = test_acc
                    state = {
                        'epoch': epoch,
                        'model': model_s.state_dict(),
                        'best_acc': best_acc,
                        'robust_acc': robust_test_acc, 
                        'mask_epoch': mask_list
                    }

                    save_file = os.path.join(opt.save_folder, '{}_stage2_best.pth'.format(opt.model_s))
                    print('saving the best model!')
                    torch.save(state, save_file)

                if attacker:
                    if robust_test_acc > best_robust_acc:
                        best_robust_acc = robust_test_acc
                        state = {
                            'epoch': epoch,
                            'model': model_s.state_dict(),
                            'best_robust_acc': best_robust_acc,
                            'natural acc': test_acc,
                            'mask_epoch': mask_list
                        }
                        
                        save_file = os.path.join(opt.save_folder, 'robust_{}_stage2_best.pth'.format(opt.model_s))
                        print('saving the best robust model!')
                        torch.save(state, save_file)


    if opt.local_rank == 0:
        if opt.use3BN or opt.use2BN:
            for val_lambda in val_lambdas:
                print('Best Accuracies:')
                print('val_lambda%s: TA: %.4f, ATA: %.4f\n' % (val_lambda, best_TA[val_lambda], best_ATA[val_lambda]))
        else:
            print('best accuracy of stage2:{}', best_acc)
            if attacker:
                print('best robust accuracy of stage2:{}', best_robust_acc)



        # save model
        state = {
            'opt': opt,
            'model': model_s.state_dict(),
            'mask_epoch': mask_list
        }
        save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
        torch.save(state, save_file)


if __name__ == '__main__':
    main()
