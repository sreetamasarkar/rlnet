import torch
from models import model_dict
import numpy as np

def load_model(model_name, model_path, n_cls):
    print('==> loading model')
    model = model_dict[model_name](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model, torch.load(model_path)['mask_epoch']

def main():
    n_cls = 10
    model_name = 'CustomResNet18'
    model_path_stage1_nat = 'save/student_model/stage1/S:CustomResNet18_T1:ResNet18_cifar10_kd_lr:0.05_r:0.5_a:0.5_b:1000.0_1_ResNet18_C10_relu82k_sensitivity_Pretrain:False/CustomResNet18_stage1_best.pth'
    model_path_stage1_adv = 'save/student_model/stage1/S:CustomResNet18_T1:ResNet18_TRADES_cifar10_a:0.5_relu82k_rslad_feat_diff_nat/robust_CustomResNet18_stage1_best.pth'
    model_nat, mask_epoch_nat = load_model(model_name, model_path_stage1_nat, n_cls)
    model_adv, mask_epoch_adv = load_model(model_name, model_path_stage1_adv, n_cls)
    nat_mask_list = [mask_epoch_nat[i].detach().cpu().numpy() for i in range(len(mask_epoch_nat))]
    adv_mask_list = [mask_epoch_adv[i].detach().cpu().numpy() for i in range(len(mask_epoch_adv))]
    intersect = [np.sum(nat_mask_list[i] * adv_mask_list[i]) for i in range(len(nat_mask_list))]
    u = [np.sum(nat_mask_list[i] | adv_mask_list[i]) for i in range(len(nat_mask_list))]
    iou = [intersect[i]/u[i] for i in range(len(nat_mask_list))]
    # mask_diff = [torch.sum(abs(mask_epoch_nat[i] - mask_epoch_adv[i])) for i in range(len(mask_epoch_nat))]
    # mask_diff = [x.item() for x in mask_diff]
    print(iou)

if __name__ == '__main__':
    main()
