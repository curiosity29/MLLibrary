import torch
from torchvision.transforms import v2
from functools import partial
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
def get_augmentation():
    # image_
    return partial(augmanetation, image_augmentation = get_image_augmentation(), image_label_augmentation = image_label_augmentation)
def augmanetation(image, label, image_augmentation, image_label_augmentation):
    image = image_augmentation(image)
    image, label = image_label_augmentation(image, label)
    return image, label
def get_image_augmentation():
    transforms = v2.Compose([
    # v2.RandomResizedCrop(size=image_shape, antialias=True),
    # v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    AddGaussianNoise(0., 0.1)
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms
def image_label_augmentation(image, label):
    stacked = torch.concat([image, label], axis = 1)
    n_channel = image.shape[1]
    transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    ])
    stacked = transforms(stacked)
    return stacked[:, :n_channel, ...], stacked[:, n_channel:, ...]
    