import numpy as np
from medpy.metric.binary import hd, dc


def txt2list(txt_path):
    with open(txt_path, 'r') as paths:
        rows = paths.readlines()
    tif_list = [row[:-1] for row in rows]
    return tif_list


def list2txt(list, txt_path):
    with open(txt_path,'w') as text:
        for item in list:
            text.write(str(item)+'\n')


def dice_coefficient(pred, gt, smooth=1e-5):
    """ computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
    N = gt.shape[0]
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    dice = (2 * intersection + smooth) / (unionset + smooth)
    return dice.sum() / N


def get_metric(pred, gt, num_class):
    '''
    compute the dice of different categories
    :param pred: math:`(N, C, H, W)`, the predict results of segmentation
    :param gt: math:`(N, 1, H, W)`, the ground truth of segmentation
    :return: DICE:
             ASDD:
    '''
    N, C, H, W = pred.shape  # torch.Size([4, 5, 256, 256])

    if gt.shape[1] != 1:
        print('ERROR')
    gt   = gt.view(N, H, W)  # torch.Size([4, 256, 256])

    pred_category = pred.argmax(dim=1)  # (N, H, W)  argmax ：https://zhuanlan.zhihu.com/p/79383099  torch.Size([4, 256, 256])

    DISC = np.zeros(num_class)  # MMWHS is 4,because is f# our kinds of classification

    # pred_category = pred_category.cpu().data.numpy()
    # gt = gt.cpu().data.numpy()

    for i in range(num_class):
        pred_i = (pred_category == (i + 1))
        gt_i   = (gt == (i + 1))

        # DISC[i] = metric.binary.dc(pred_i, gt_i)
        # ASSD[i] = metric.binary.assd(pred_i, gt_i)
        DISC[i] = dice_coefficient(pred_i, gt_i, smooth=1e-5)

    return DISC


