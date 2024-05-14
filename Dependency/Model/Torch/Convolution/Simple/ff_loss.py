import torch
import torch.nn as nn
from functools import partial
from torchvision.ops import sigmoid_focal_loss
from torch.nn import BCEWithLogitsLoss
from monai.losses import DiceFocalLoss


bce_loss = nn.BCELoss(reduction='mean')
focal_loss = partial(sigmoid_focal_loss, reduction = "mean", alpha = -1)
# ce_loss = nn.CrossEntropyLoss(weights = class_weights, reduction = "mean")
dice_focal_loss = DiceFocalLoss(
    include_background = True, to_onehot_y = False, 
    sigmoid = False, softmax = False, 
    jaccard = False,
    reduction = "mean",
    weight = None,
    gamma = 2.0,
)
bce_dice_folcal_loss = lambda x: dice_focal_loss(x) + bce_loss(x)


def loss_calc(d0, labels_v, base_loss = bce_loss):

            # base_loss = lambda x: 
            

    loss = base_loss(d0,labels_v)

    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss


def get_loss(base_loss = "bce", loss_args = {}, device = "cuda"):
    match base_loss:
        case "bce":
            base_loss = bce_loss
        case "bce_logit":
            base_loss = BCEWithLogitsLoss(reduction = loss_args["reduction"], pos_weight= torch.tensor(loss_args["pos_weight"]).view(1,1,1).to(device))
        case "focal":
            base_loss = focal_loss
        # case "ce":
        #     base_loss = nn.CrossEntropyLoss(weights = class_weights, reduction = "mean")
        case "dice_focal":
            base_loss =  dice_focal_loss
        case "bce_dice_focal":
            base_loss = bce_dice_folcal_loss

    return partial(loss_calc, base_loss = base_loss)