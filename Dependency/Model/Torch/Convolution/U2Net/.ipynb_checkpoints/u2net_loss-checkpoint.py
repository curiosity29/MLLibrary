import torch.nn as nn
from functools import partial
from torchvision.ops import sigmoid_focal_loss

# bce_loss = nn.BCELoss(reduction='mean')

from monai.losses import DiceFocalLoss

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v, weights = [1.] * 7 , base_loss = "bce", class_weights = None):
    match base_loss:
        case "bce":
            base_loss = nn.BCELoss(reduction='mean')
        case "focal":
            base_loss = partial(sigmoid_focal_loss, reduction = "mean", alpha = -1)
        case "ce":
            base_loss = nn.CrossEntropyLoss(weights = class_weights, reduction = "mean")
        case "dice_focal":
            df_loss = DiceFocalLoss(
                include_background = False, to_onehot_y = False, 
                sigmoid = False, softmax = False, 
                jaccard = False,
                reduction = "mean",
                weight = None,
                gamma = 2.0,
            )
            base_loss =  df_loss
            
        
    loss0 = base_loss(d0,labels_v) * weights[0]
    loss1 = base_loss(d1,labels_v) * weights[1]
    loss2 = base_loss(d2,labels_v) * weights[2]
    loss3 = base_loss(d3,labels_v) * weights[3]
    loss4 = base_loss(d4,labels_v) * weights[4]
    loss5 = base_loss(d5,labels_v) * weights[5]
    loss6 = base_loss(d6,labels_v) * weights[6]

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss0, loss