'''
消融pipeline
xv
'''
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from tqdm import tqdm
import config
import DataLoad_flexible as DataLoad
# import DataLoad_verse20
import new_model_structure.Unet_Model_updata as SaNet_demo
import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import torch

def one_hot_encoding(input_tensor, num_classes):
    B, _, D, H, W = input_tensor.shape
    one_hot_tensor = torch.zeros(B, num_classes, D, H, W, dtype=torch.float32, device=input_tensor.device)
    one_hot_tensor = one_hot_tensor.scatter_(1, input_tensor, 1.0)
    return one_hot_tensor

def Dice_loss(pred, target, num_classes):
    """
    这个函数保持不变，它正确地计算了单个预测张量的Dice Loss。
    注意：为了计算效率和逻辑清晰，我稍微修改了one-hot编码的处理方式。
    """
    pred_probs = F.softmax(pred, dim=1)  # (B, C, D, H, W)
    target_one_hot = one_hot_encoding(target, num_classes)  # (B, C, D, H, W)

    total_dice_loss = 0.0
    # 从 class_idx = 1 开始循环，通常背景类别（0）不参与Dice Loss计算
    for class_idx in range(1, num_classes):
        pred_class = pred_probs[:, class_idx, ...]
        target_class = target_one_hot[:, class_idx, ...]

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        eps = 1e-6
        dice = (2 * intersection + eps) / (union + eps)
        total_dice_loss += (1 - dice)

    # 对类别取平均
    mean_dice_loss = total_dice_loss / (num_classes - 1)
    return mean_dice_loss


def deep_supervision_dice_loss(preds, target, num_classes, weights=None):
    """
    计算深度监督模型的总Dice Loss。

    Args:
        preds (list of Tensors): 模型的输出，一个包含多个预测张量的列表。
        target (Tensor): 真实的标签。
        num_classes (int): 类别数。
        weights (list of float, optional): 每个预测输出的权重。
                                           默认为 [1.0, 0.6, 0.4] 或类似形式。
    """
    if not isinstance(preds, list):
        # 如果因为某些原因（例如在验证阶段关闭了深度监督），输入不是列表，
        # 则直接计算损失。
        return Dice_loss(preds, target, num_classes)

    if weights is None:
        # 定义默认权重。通常主输出权重最高，辅助输出权重递减。
        # 权重数量应与preds列表的长度匹配。
        # 这里的权重设置是经验性的，你可以根据你的任务进行调整。
        weights = [1.0] + [0.6 / (2 ** i) for i in range(len(preds) - 1)]

    if len(weights) != len(preds):
        raise ValueError(f"The number of weights ({len(weights)}) must match the number of predictions ({len(preds)}).")

    total_loss = 0.0
    for i, pred in enumerate(preds):
        # 注意：这里的target可能需要根据pred的尺寸进行下采样，
        # 但在你的代码中，辅助输出已经上采样到了原始尺寸，所以target不需要调整。
        # 如果你的模型输出是多尺度的，你需要在这里对target进行降采样以匹配pred的尺寸。
        loss = Dice_loss(pred, target, num_classes)
        total_loss += (loss * weights[i])

    return total_loss

def soft_skeletonize(x, iterations=10):
    x = x.float()
    """Soft differentiable skeletonization"""
    for _ in range(iterations):
        # print("Tensor shape after modification:", x.shape)  # 打印修改后的形状
        # 改3D卷积
        min_pool = -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)
        contour = F.relu(min_pool - x)
        x = F.relu(x - contour)
    return x

def soft_cldice(pred, gt, smooth=1e-6):
    skel_pred = soft_skeletonize(pred)
    skel_gt = soft_skeletonize(gt)

    tprec = (skel_pred * gt).sum() / (skel_pred.sum() + smooth)
    tsens = (skel_gt * pred).sum() / (skel_gt.sum() + smooth)

    return 1 - 2 * tprec * tsens / (tprec + tsens + smooth)
'''
def compute_mean_dice_and_iou_no_one_hot(pred, target, num_classes):
    """
    计算多目标分割任务的平均 DICE 系数和 IOU，支持 B, C, D, H, W 格式，标签为整型索引格式。

    Args:
        pred (torch.Tensor): 网络输出 (B, C, D, H, W)，未经过 softmax。
        target (torch.Tensor): 目标标签 (B, D, H, W)，整型索引格式。
        num_classes (int): 分类的总数。

    Returns:
        dict: 包含平均 DICE 和 IOU 的字典，结构为 {"mean_dice": float, "mean_iou": float}。
    """
    # 将网络输出通过 softmax 转化为概率
    pred_probs = F.softmax(pred, dim=1)  # (B, C, D, H, W)

    # 初始化累积 DICE 和 IOU
    total_dice = 0.0
    total_iou = 0.0

    # 将标签转换为 one-hot 格式
    # target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
    target_one_hot = one_hot_encoding(target.to(torch.int64), num_classes)
    for class_idx in range(num_classes):
        # 取出当前类别的预测和目标
        pred_class = pred_probs[:, class_idx, ...]  # (B, D, H, W)
        target_class = target_one_hot[:, class_idx, ...]  # (B, D, H, W)

        # 计算交集和并集
        intersection = (pred_class * target_class).sum()  # 针对每个 batch 求和
        union = pred_class.sum() + target_class.sum()
        iou_denominator = union - intersection  # IOU 的分母

        # 防止数值错误
        eps = 1e-6

        # 计算当前类别的 DICE 系数和 IOU
        dice = (2 * intersection + eps) / (union + eps)  # DICE 系数
        iou = (intersection + eps) / (iou_denominator + eps)  # IOU

        # 累加每个类别的平均 DICE 和 IOU（对 batch 求均值后累加）
        total_dice += dice.mean()
        total_iou += iou.mean()

    # 计算平均 DICE 和 IOU
    mean_dice = total_dice / num_classes
    mean_iou = total_iou / num_classes

    return {"mean_dice": mean_dice.item(), "mean_iou": mean_iou.item()}
    '''
def compute_mean_dice_and_iou_for_foreground(pred, target, num_classes):
    """
    计算多目标分割任务的平均DICE和IOU，只针对前景类别。
    """
    # 如果只有一个类别（通常是二分类，0背景1前景），我们依然只关心前景。
    if num_classes <= 1:
        # 这种情况下，此多类别函数可能不适用，或者需要特殊处理
        # 为避免除以0，直接返回0或调用二分类的函数
        return {"mean_dice": 0.0, "mean_iou": 0.0}

    pred_probs = F.softmax(pred, dim=1)
    target_one_hot = one_hot_encoding(target.to(torch.int64), num_classes)

    total_dice = 0.0
    total_iou = 0.0
    
    # 【关键修改1】: 循环从类别 1 开始，跳过背景
    for class_idx in range(1, num_classes):
        pred_class = pred_probs[:, class_idx, ...]
        target_class = target_one_hot[:, class_idx, ...]

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        iou_denominator = union - intersection

        eps = 1e-6
        dice = (2 * intersection + eps) / (union + eps)
        iou = (intersection + eps) / (iou_denominator + eps)

        total_dice += dice
        total_iou += iou

    # 【关键修改2】: 除以总的前景类别数量
    mean_dice = total_dice / (num_classes - 1)
    mean_iou = total_iou / (num_classes - 1)

    return {"mean_dice": mean_dice.item(), "mean_iou": mean_iou.item()}

def writeFile(name,info):
    with open(name, "a+") as f:
        f.write(str(info))
        f.write("\n")

def train(net,device,root_data_path):
    optimizer = optim.Adam(net.parameters(), lr=config.lr,betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=200,  # T_0就是初始restart的epoch数目
        T_mult=1,  # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 * T_mult
        eta_min=1e-6  # 最低学习率
    )
    data_all_path = [os.path.join(root_data_path,f) for f in os.listdir(root_data_path)]
    print(f"共发现 {len(data_all_path)} 个样本")
    random.seed(42)  # 保证可重复
    train_data_path = []
    train_root = os.path.join(root_data_path, 'train')
    for category in os.listdir(train_root):
        category_path = os.path.join(train_root, category)
        if os.path.isdir(category_path):
            for sample in os.listdir(category_path):
                sample_path = os.path.join(category_path, sample)
                if os.path.isdir(sample_path):
                    train_data_path.append(sample_path)

    val_data_path = []
    val_root = os.path.join(root_data_path, 'val')
    for category in os.listdir(val_root):
        category_path = os.path.join(val_root, category)
        if os.path.isdir(category_path):
            for sample in os.listdir(category_path):
                sample_path = os.path.join(category_path, sample)
                if os.path.isdir(sample_path):
                    val_data_path.append(sample_path)
    
    print(f"共发现 {len(train_data_path)} 个训练样本")
    print(f"共发现 {len(val_data_path)} 个验证样本")
    train_data = DataLoad.DataSetUtil(train_data_path)
    val_data = DataLoad.DataSetUtil(val_data_path)
    train_dataLoader = DataLoader(train_data, config.batch_size, shuffle=True, num_workers=0)
    val_dataLoader = DataLoader(val_data, config.batch_size, shuffle=False, num_workers=0)
    best_loss = 1
    best_iou = 0
    net.train()
    for epoch in range(config.epochs):
        print("current epoch is:{}".format(epoch))
        train_loss_set = []
        # train model
        alpha = np.clip(1.0 * (epoch + 1) / config.epochs, 0.01, 1)
        for img_data, label in tqdm(train_dataLoader):
            optimizer.zero_grad()
            # move data to gpu device
            img_data = img_data.to(device,dtype = torch.float32)
            label_data = label.to(device, dtype = torch.int64)
            # forward pass
            pred_out = net(img_data)
            # compute loss
            loss = deep_supervision_dice_loss(pred_out, label_data, num_classes=2,weights=[1.0, 0.5, 0.25])
            # recording loss
            train_loss_set.append(loss.item())
            # backward pass
            loss.backward()
            # update parameters
            optimizer.step()
        scheduler.step()
        # validate model
        val_loss_set = []
        val_dice_set = []
        val_iou_set = []
        net.eval()  # <--- 【核心修正】在开始验证前，将模型设置为评估模式
        with torch.no_grad():
            for vail_data, vail_label in val_dataLoader:
                vail_data = vail_data.to(device, dtype=torch.float32)
                vail_label = vail_label.to(device, dtype=torch.int64)
                vail_pred = net(vail_data)
                # val_loss = focal_loss(vail_pred, vail_label)
                val_loss = Dice_loss(vail_pred, vail_label,num_classes=2)

                iou_dice = compute_mean_dice_and_iou_for_foreground(vail_pred,vail_label,2)
                val_iou = iou_dice["mean_iou"]
                val_dice = iou_dice["mean_dice"]
                val_loss_set.append(val_loss.item())
                val_iou_set.append(val_iou)
                val_dice_set.append(val_dice)

            val_loss_set = np.array(val_loss_set)
            val_iou_set = np.array(val_iou_set)
            val_dice_set = np.array(val_dice_set)
            if np.mean(val_loss_set) < best_loss:
                # choose best model to save
                torch.save(net.state_dict(), r'ModelSet_new2_less_train/xv_930_flexible_3.pth')
                best_loss = np.mean(val_loss_set)
            if np.mean(val_iou_set) > best_iou:
                torch.save(net.state_dict(), r'ModelSet_new2_less_train/xv_930_flexible_3.pth')
                best_iou = np.mean(val_iou_set)
            print(f"epoch:{epoch} train loss:{np.mean(train_loss_set)}  val_loss:{np.mean(val_loss_set)} val_iou:{np.mean(val_iou_set)} val_dice:{np.mean(val_dice_set)}")
        #save this epoch mean loss
            writeFile(r"loss1_new2_less_train/xv_930_flexible_3_train_loss"  + ".txt", np.mean(train_loss_set))
            writeFile(r"loss1_new2_less_train/xv_930_flexible_3_val_loss"  + ".txt", np.mean(val_loss_set))
            writeFile(r"loss1_new2_less_train/xv_930_flexible_3_val_iou" + ".txt", np.mean(val_iou_set))
            writeFile(r"loss1_new2_less_train/xv_930_flexible_3_val_dice" + ".txt", np.mean(val_dice_set))

if __name__ == '__main__':
    '''
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 torch_ddp.py
    '''

    print("this is trainning")
    root_data_path = config.data_path
    
    channel_num = [64, 128, 256, 512, 512] # list_ch 长度为5，最后一个给瓶颈层
    alpha_list = [0.90, 0.90, 0.80, 0.75, 0.75] # alpha_list 长度为5,代表全局
    # alpha_list = [0.1, 0.1, 0.2, 0.25, 0.25] ##通常浅层看局部纹理，深层看全局语义
    net = SaNet_demo.FSSANet_V3(list_ch=channel_num,
        alpha_list=alpha_list,
        in_channels=1, 
        out_channels=2,
        deep_supervision=True 
    )
    device = torch.device("cuda")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device_ids = [0, 1]
    net.to(device)
    if config.is_load_pre_model:
        print("load pre_train model")
        model_path = r'/home/yons/xv/net/ModelSet2/best_iou_all_data_last_struct.pth'
        net.load_state_dict(torch.load(model_path), strict=False)
        print("load pre_train model:",model_path)
    if config.number_gpu > 1:
        net = nn.DataParallel(net,device_ids=device_ids)
    train(net,device,root_data_path)
