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

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample

from helper.util import adjust_learning_rate

#SK:added the GRAMM loss to the distill zoo and imported here
from distiller_zoo import DistillKL, Attention
from helper.loops import train_distill as train, validate
from helper.attack import PGD, FGSM
from helper.pretrain import init

from admm_core import Masking, CosineDecay, LinearDecay, add_sparse_args

from einops import rearrange, reduce, repeat


def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--use_l2_norm', type=bool, default=False, help='choose mask difference calculation function')
    parser.add_argument('--use_pruning', type=bool, default=False, help='choose whether to use pruning in second stage')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--t1_epochs', type=int, default=200, help='number of first teacher epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,150,180', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar100', 'cifar10'], help='dataset')

    # model
    # The  resnet50,38,26 matches the models of SASA paper and
    # are written by us. The ResNet50 model is of original CRD paper
    parser.add_argument('--model_s', type=str, default='Resnet18',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_22_8', 'wrn_40_1', 'wrn_40_2',
                                 'resnet50', 'resnet38', 'resnet26',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet18', 'ResNet18Prelu', 'ResNet34',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'CustomResNet18'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention',
                                                                    'similarity','correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

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
    parser.add_argument('--robust_train_mode', type=str, default='pgd', choices=['trades', 'rslad', 'pgd'])

    parser.add_argument("--seed", help="set random generator seed", default='0')

    add_sparse_args(parser)

    opt = parser.parse_args()
    torch.manual_seed(int(opt.seed))
    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = opt.model_s
    # opt.model_name = 'S:{}_T:{}_{}_{}_lr:{}_r:{}_a:{}_b:{}_{}_dense:{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
    #                                                     opt.learning_rate, opt.gamma, opt.alpha, opt.beta, opt.trial, opt.dense)
    opt.model_name = 'S:{}_T:{}_{}_lr:{}_a:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, 
                                                        opt.learning_rate, opt.alpha, opt.robust_train_mode)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def load_teacher(opt, model_path, n_cls):
    print('==> loading teacher model')
    model_t = opt.model_t
    model = model_dict[model_t](num_classes=n_cls)
    state_dict = torch.load(model_path)['model']
    model.load_state_dict(state_dict)
    print('==> done')
    return model

def main():
    best_acc = 0
    best_robust_acc = 0

    opt = parse_option()
    print(opt)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    
    attacker = None
    if opt.attack_mode == 'pgd':
        if opt.robust_train_mode == 'rslad' or opt.robust_train_mode == 'trades':
            attacker = PGD(eps=opt.eps, steps=opt.steps, criterion=criterion_div)
        else:
            attacker = PGD(eps=opt.eps, steps=opt.steps)
    elif opt.attack_mode == 'fgsm': attacker = FGSM(eps=opt.eps)

   # Validate using PGD-20 
    attacker_val = PGD(eps=opt.eps, steps=20)

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 100
    elif opt.dataset == 'cifar10':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar10_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True, attacker=attacker)
        n_cls = 10
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt, opt.path_t, n_cls)
    if opt.self_attn == True:
        print("Choosing self attention model as student.")
        model_s = model_dict[opt.model_s](num_classes=n_cls, stem=opt.stem)
    else:
        model_s = model_dict[opt.model_s](num_classes=n_cls)

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

    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, teacher_acc_robust, _, _ = validate(val_loader, model_t, criterion_cls, opt, attacker=attacker_val)
    print('teacher accuracy: ', teacher_acc)
    if attacker:
        print('robust teacher accuracy: ', teacher_acc_robust)
        
    # routine
    for epoch in range(1, opt.epochs + 1):
        # if epoch == opt.t1_epochs + 1:
            
        #     if not opt.dense:
        #         if opt.decay_schedule == 'cosine':
        #             decay = CosineDecay(opt.prune_rate, len(train_loader)*(opt.epochs))
        #         elif opt.decay_schedule == 'linear':
        #             decay = LinearDecay(opt.prune_rate, len(train_loader)*(opt.epochs))
        #         print('using {} decay schedule'.format(opt.decay_schedule))

        #         mask_weight = Masking(optimizer, decay, prune_rate=opt.prune_rate, prune_mode=opt.prune, \
        #             growth_mode=opt.growth, redistribution_mode=opt.redistribution, verbose=opt.verbose)
        #         mask_weight.add_module(model_s, density=opt.density)


        # use_model_t_2 = False
        # if epoch > opt.t1_epochs: break
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, robust_train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt, attacker=attacker)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        if attacker:
            logger.log_value('robust_train_acc', robust_train_acc, epoch)
   
        test_acc, robust_test_acc, test_loss, _ = validate(val_loader, model_s, criterion_cls, opt, attacker=attacker_val)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        if attacker:
            logger.log_value('robust_test_acc', robust_test_acc, epoch)

        # save the best model
        # if epoch <= opt.t1_epochs:
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }

            save_file = os.path.join(opt.save_folder, '{}_student_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        if attacker:
            if robust_test_acc > best_robust_acc:
                best_robust_acc = robust_test_acc
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'best_robust_acc': best_robust_acc,
                    'natural acc': test_acc
                }

                save_file = os.path.join(opt.save_folder, 'robust_{}_student_best.pth'.format(opt.model_s))
                print('saving the best robust model!')
                torch.save(state, save_file)

    print('best student accuracy:{}', best_acc)
    if attacker:
        print('best robust accuracy of stage2:{}', best_robust_acc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_student_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
