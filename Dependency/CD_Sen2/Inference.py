import torch
from torch.autograd import Variable
from functools import partial
import numpy as np

def get_predictor(model, mode = "normal"):
    if mode == "swapped":
        return partial(swapped_predict, model = model, n_channel = 4)

    return partial(predict, model = model)


def swapped_predict(batch, model, n_channel = 4):
    batch = np.concatenate((batch[..., n_channel:], batch[..., :n_channel]), axis = -1)
    return predict(batch, model)
    

def predict(batch, model):
    batch = np.transpose(batch, (0, 3, 1, 2))
    
    model.eval()
    with torch.no_grad():
        batch = torch.from_numpy(batch).type(torch.FloatTensor)  
            # wrap them in Variable
        batch_cuda = Variable(batch.cuda(), requires_grad=False)
        pred = model(batch_cuda)[0].cpu().detach().numpy()
        del batch_cuda
    pred = np.transpose(pred, (0, 2, 3, 1))
    # pred = np.where(pred > 0.5, 1, 0)
    return pred