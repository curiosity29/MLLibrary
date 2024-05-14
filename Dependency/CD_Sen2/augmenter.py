import torch
from torchvision.transforms import v2
from functools import partial

import satellite_cloud_generator as scg

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p = 0.5):
        self.std = std
        self.mean = mean
        self.p = p
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddCloud(object):
    def __init__(self, min_lvl=0., max_lvl=1., n_channel = 4, p=0.5):
        self.min_lvl = min_lvl
        self.max_lvl = max_lvl
        self.n_channel = n_channel
        self.p = p
        # self.std = std
        # self.mean = mean
        
    def __call__(self, tensor):
        
        if torch.rand(1) > self.p:
            return tensor

        
        image, label = tensor[:, :self.n_channel * 2, ...], tensor[:, self.n_channel * 2:, ...]
        
        if torch.rand(1)> 0.5: ## cloud on first image
            image[:, :self.n_channel, ...], cmask, smask  = scg.add_cloud_and_shadow(image[:, :self.n_channel, ...],
                                     min_lvl= self.min_lvl,
                                     max_lvl= self.max_lvl,
                                     return_cloud=True
                                 )
        else: ## cloud on second image
            image[:, -self.n_channel:, ...], cmask, smask  = scg.add_cloud_and_shadow(image[:, -self.n_channel:, ...],
                                     min_lvl= self.min_lvl,
                                     max_lvl= self.max_lvl,
                                     return_cloud=True
                                 )
    
        # turn cloud mask to background
        # ignore shadow mask
        # print(cmask.shape)
        cmask, _ = torch.max(cmask, 1, keepdim = True)
        cmask = torch.where(cmask > 0.4, 0., 1.) # invert cloud mask
        # print(cmask.shape)
        # print(label.shape)
        ##
        # label = torch.from_numpy(label)
        # cmask[0, ...]
        ##
        # print(label.shape)
        # print(cmask.shape)
        label = label * cmask
    
        # return image, label
        return torch.concat([image, label], axis = 1)

    
    def __repr__(self):
        return self.__class__.__name__ + '(min_lvl={0}, max_lvl={1}, n_channel={2})'.format(self.min_lvl, self.max_lvl, self.n_channel)

    
def get_augmentation(flip = True, noise_std = 0.05, cloud_prob = 0.2):
    # image_
    return partial(augmentation, 
                   image_augmentation = get_image_augmentation(noise_std = noise_std), 
                   image_label_augmentation = partial(image_label_augmentation, cloud_prob = cloud_prob, flip = flip))
def augmentation(image, label, image_augmentation, image_label_augmentation):
    image = image_augmentation(image)
    image, label = image_label_augmentation(image, label)
    return image, label
def get_image_augmentation(noise_std = 0.1):
    transforms = v2.Compose([
    # v2.RandomResizedCrop(size=image_shape, antialias=True),
    # v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    AddGaussianNoise(mean = 0., std = noise_std, p = 0.5)
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms
def image_label_augmentation(image, label, cloud_prob = 0.5, flip = True):
    if flip:
        flip_prob = 0.5
    else:
        flip_prob = 0.
    stacked = torch.concat([image, label], axis = 1)
    n_channel = image.shape[1]
    transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=flip_prob),
    v2.RandomVerticalFlip(p=flip_prob),
    AddCloud(min_lvl=0., max_lvl=1., n_channel = 4, p = cloud_prob),
    ])
    stacked = transforms(stacked)
    return stacked[:, :n_channel, ...], stacked[:, n_channel:, ...]

# def add_cloud(image, label):
#     image, cmask, smask  = scg.add_cloud_and_shadow(image,
#                              min_lvl=0.0,
#                              max_lvl=1.0,
#                              return_cloud=True
#                          )

#     # turn cloud mask to background
#     # ignore shadow mask
#     cmask = np.where(cmask > 0.3, 0., 1.) # invert cloud mask
#     label = label * cmask

#     return image, label
    

    

    
    