from __future__ import print_function

import argparse
import socket

import torch, copy
import matplotlib.pyplot as plt
import numpy as np

from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders
from einops import repeat


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')

    # models (note there are two resnet50s, one from actual CRD paper,
    # the other written by us to match the param count of Stand-alone SA (SASA) paper
    # The resnet50, resnet38, resnet26 are matching the model of SASA paper
    parser.add_argument('--modelAR', type=str, default='ResNet18_plot',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'ResNet18', 'ResNet34', 'ResNet50', 'wrn_16_1', 'wrn_16_2', 
                                 'wrn_40_1', 'wrn_40_2', 'resnet50', 'resnet38', 'resnet26',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'CustomResNet18', 'ResNet18_plot', 'CustomResNet18_plot'])

    parser.add_argument('--modelPR', type=str, default='CustomResNet18_plot',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'ResNet18', 'ResNet34', 'ResNet50', 'wrn_16_1', 'wrn_16_2', 
                                 'wrn_40_1', 'wrn_40_2', 'resnet50', 'resnet38', 'resnet26',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'CustomResNet18', 'ResNet18_plot', 'CustomResNet18_plot'])
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10'], help='dataset')


    # settings for norm layers
    # Reference: https://github.com/EkdeepSLubana/BeyondBatchNorm/blob/main/main.py
    parser.add_argument("--norm_type", help="Normalization layer to be used", default='BatchNorm', choices=['Plain', 'BatchNorm', 'LayerNorm', 'Instance Normalization', 'GroupNorm', 'Filter Response Normalization', 'Weight Normalization', 'Scaled Weight Standardization', 'EvoNormSO', 'EvoNormBO', 'Variance Normalization', 'Mean Centering'])
    parser.add_argument("--p_grouping", help="Number of channels per group for GroupNorm", default='32', choices=['1', '0.5', '0.25', '0.125', '0.0625', '0.03125', '0.0000001', '8', '16', '32', '64'])
    parser.add_argument("--conv_type", help="Convolutional layer to be used", default='Plain', choices=['Plain', 'sWS', 'WeightNormalized', 'WeightCentered'])
    parser.add_argument("--probe_layers", help="Probe activations/gradients?", default='True', choices=['True', 'False'])
    parser.add_argument("--cfg", help="Model configuration", default='cfg_10')
    parser.add_argument("--skipinit", help="Use skipinit initialization?", default='False', choices=['True', 'False'])
    parser.add_argument("--preact", help="Use preactivation variants for ResNet?", default='False', choices=['True', 'False'])

    parser.add_argument("--seed", help="set random generator seed", default='0')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_option()
    torch.manual_seed(int(opt.seed))
    # dataloader
    if opt.dataset == 'cifar100':
        _, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    elif opt.dataset == 'cifar10':
        _, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
    else:
        raise NotImplementedError(opt.dataset)

    # AR model
    modelAR = model_dict[opt.modelAR](num_classes=n_cls, conv_type=opt.conv_type, norm_type=opt.norm_type , p_grouping=float(opt.p_grouping), skipinit=opt.skipinit)
    state_dictAR = torch.load('save/models/ResNet18_BN_best.pth')
    state_dictAR = state_dictAR['model']
    modelAR.load_state_dict(state_dictAR)
    modelAR = modelAR.cuda()
    modelAR.eval()

    # PR model
    modelPR = model_dict[opt.modelPR](num_classes=n_cls, conv_type=opt.conv_type, norm_type=opt.norm_type , p_grouping=float(opt.p_grouping), skipinit=opt.skipinit)
    modelPR_path = 'save/student_model/stage2/GhostMask_S:CustomResNet18_T1:ResNet18_cifar100_kd_lr:0.01_r:0.1_a:0.9_b:1000.0_1_stem:False_dense:True_stage1:0_NormType:BatchNorm/CustomResNet18_stage1_best.pth'
    state_dictPR = torch.load(modelPR_path)
    mask_list = state_dictPR['mask_epoch']

    state_dictPR = state_dictPR['model']
    modelPR.load_state_dict(state_dictPR)
    modelPR = modelPR.cuda()
    modelPR.eval()

    postact_AR, postact_PR = dict(), dict()
    for i in range(17): 
        postact_AR[i] = []
        postact_PR[i] = []

    mask_list_copy = copy.deepcopy(mask_list)
    for idx, (input, _) in enumerate(val_loader):
        if idx == 10: break
        input = input.float()
        if torch.cuda.is_available(): input = input.cuda()

        featuresAR, featuresPR = [], []
        # ===================forward=====================
        _, _ = modelAR(input, featuresAR, postact_AR)

        for mask_index, _ in enumerate(mask_list):
            mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input.shape)[0]) # b c h w
        _, _ = modelPR(input, mask_list, featuresPR, postact_PR)
        mask_list = copy.deepcopy(mask_list_copy)

    sparsity_AR, sparsity_PR = [], []

    for _,layermap in postact_AR.items():
        N,C,H,W = layermap[0].shape
        num_elem = N*C*H*W
        s = 0
        for map in layermap: s += (num_elem-(torch.count_nonzero(map)).cpu())
        sparsity_AR.append(s/10/num_elem)

    for _,layermap in postact_PR.items():
        N,C,H,W = layermap[0].shape
        num_elem = N*C*H*W
        s = 0
        for map in layermap: s += (num_elem-(torch.count_nonzero(map)).cpu())
        sparsity_PR.append(s/10/num_elem)

    barWidth = 0.25
        
    # Set position of bar on X axis
    br1 = np.arange(17)
    br2 = [x + barWidth for x in br1]
    
    # Make the plot
    plt.bar(br1, sparsity_AR, color ='r', width = barWidth,
            edgecolor ='grey', label ='AR')
    plt.bar(br2, sparsity_PR, color ='b', width = barWidth,
            edgecolor ='grey', label ='PR_82k')

    # Adding Xticks
    plt.xlabel('Post-activation Layer', fontweight ='bold', fontsize = 15)
    plt.ylabel('sparsity', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth/2 for r in range(17)],np.arange(1,18))

    lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='best')
    plt.savefig('sparsity.png',bbox_extra_artists=(lg,), bbox_inches='tight')

if __name__ == '__main__':
    main()