import torch
import torch.nn as nn
from functools import partial
from torchvision.ops import sigmoid_focal_loss
from monai.losses import DiceFocalLoss
from torch.nn import BCEWithLogitsLoss



# ce_loss = nn.CrossEntropyLoss(weights = class_weights, reduction = "mean")
# dice_focal_loss = DiceFocalLoss(
#     include_background = True, to_onehot_y = False, 
#     sigmoid = False, softmax = False, 
#     jaccard = False,
#     reduction = "mean",
#     weight = None,
#     gamma = 2.0,
# )


def get_loss(base_loss = "bce", loss_args = {}, class_weights = None, device = "cuda"):
    match base_loss:
        case "bce":
            base_loss = nn.BCELoss(reduction='mean')
        case "sigmoid_focal":
            base_loss = sigmoid_focal_loss#partial(sigmoid_focal_loss, reduction = "mean", alpha = -1)
        # case "ce":
        #     base_loss = nn.CrossEntropyLoss(weights = class_weights, reduction = "mean")
        
        case "bce_logit":
            base_loss = BCEWithLogitsLoss(reduction=loss_args["reduction"], pos_weight=torch.tensor(1.).view(1,1,1).to(device))

        ##
        case "dice_focal":
            base_loss =  DiceFocalLoss(
                    include_background = True, to_onehot_y = False, 
                    sigmoid = False, softmax = False, 
                    jaccard = False,
                    reduction = "mean",
                    weight = None,
                    gamma = 2.0,
                )
        # case "bce_dice_focal":
            # bce_dice_folcal_loss = lambda x: dice_focal_loss(x) + bce_loss(x)
            # base_loss = bce_dice_folcal_loss
            # base_loss = lambda x: 
    return partial(loss_calc, base_loss = base_loss)
        

def loss_calc(d0, d1, d2, d3, d4, d5, d6, labels_v, weights = [1.] * 7, base_loss = None):
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