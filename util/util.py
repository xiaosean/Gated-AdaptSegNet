"""
Modify from https://github.com/NVlabs/MUNIT/blob/master/utils.py
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from PIL import Image
import yaml
import numpy as np
# from torch.utils.serialization import load_lua
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
from torch.optim import lr_scheduler
# from torchvision import transforms
import torch
import os
import math
# import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.init as init
from torch.utils import data

from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import SYNYHIADataSet
from dataset.cityscapes_dataset import cityscapesDataSet

# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_data_loader_folder    : folder-based data loader
# get_config                : load yaml file
# eformat                   :
# write_2images             : save output image
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# write_one_row_html        : write one row of the html file for output images
# write_html                : create the html file.
# write_loss
# slerp
# get_slerp_interp
# get_model_list
# load_vgg16
# vgg_preprocess
# get_scheduler
# weights_init

# TODO:data arguments
def get_all_data_loaders(conf):
    num_steps = conf['num_steps']
    iter_size = conf['iter_size']
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    random_scale_opt = conf["random_scale"]
    mirror_opt = conf["random_mirror"]
    random_seed = conf["random_seed"]
    # image resize function (w, h)
    dataset_name = conf["source_dataset"]
    GTA_size = (conf["input_size_w"], conf["input_size_h"])
    City_size = (conf["input_target_size_w"], conf["input_target_size_h"])
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    subset_ratio = 1
    if conf["data_subset"]:
        subset_ratio = conf["data_subset"]
        print("training dataset use subset_ratio =", subset_ratio)
    print("GTA_size input size =", GTA_size)
    print("City_size input_size_target =", City_size)
    # source domain data
    # todo: use num_worker will increase gpu memory maybe a issue
    # train_loader = data.DataLoader(
    #     GTA5DataSet(conf['data_directory'], conf['data_list_path'], max_iters=num_steps * iter_size * batch_size,
    #                 crop_size=GTA_size,
    #                 scale=random_scale_opt, mirror=mirror_opt, mean=IMG_MEAN),
    #     batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    assert dataset_name == "GTA5" or dataset_name == "SYNTHIA"
    if dataset_name == "GTA5":
        train_loader = data.DataLoader(
            GTA5DataSet(conf['data_directory'], conf['data_list_path'], max_iters=num_steps * iter_size * batch_size,
                        crop_size=GTA_size,
                        scale=random_scale_opt, mirror=mirror_opt, mean=IMG_MEAN, subset_ratio=subset_ratio, random_seed=random_seed),
            batch_size=batch_size, shuffle=True, pin_memory=True)
    elif dataset_name == "SYNTHIA":
        train_loader = data.DataLoader(
            SYNYHIADataSet(conf['data_directory'], conf['data_list_path'], max_iters=num_steps * iter_size * batch_size,
                        crop_size=GTA_size,
                        scale=random_scale_opt, mirror=mirror_opt, mean=IMG_MEAN),
            batch_size=batch_size, shuffle=True, pin_memory=True)

    # target domain data
    # target_loader = data.DataLoader(
    #     cityscapesDataSet(conf['data_directory_target'], conf['data_list_path_target'],
    #                 max_iters=num_steps * iter_size * batch_size,
    #                 crop_size=City_size,
    #                 scale=random_scale_opt, mirror=mirror_opt, mean=IMG_MEAN, set="train"),
    #     batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    target_loader = data.DataLoader(
        cityscapesDataSet(conf['data_directory_target'], conf['data_list_path_target'],
                          max_iters=num_steps * iter_size * batch_size,
                          crop_size=City_size,
                          scale=False, mirror=mirror_opt, mean=IMG_MEAN, set="train"),
        batch_size=batch_size, shuffle=True, pin_memory=True)

    return train_loader, target_loader

def get_cityscapes_valid_dataloader():
    DATA_DIRECTORY = './data/Cityscapes/data'
    DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    SET = 'val'
    testloader = data.DataLoader(cityscapesDataSet(DATA_DIRECTORY, DATA_LIST_PATH, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=SET),
                                    batch_size=1, shuffle=False, pin_memory=True)
    return testloader

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def open_Image(path):
    return Image.open(path)

def eformat(f, prec):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d"%(mantissa, int(exp))


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]

def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)

    return init_fun

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def paint_predict_image_np(predict_image):
   return np.asarray(paint_predict_image(predict_image))

def paint_predict_image(predict_image):
    """input model's output image it will paint color """
    # ===============for colorize mask==============
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    def colorize_mask(mask):
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(palette)

        return new_mask

    def output_to_image(output):
        # input
        # ------------------
        #   G's output feature map :(c, w, h, num_classes)
        #
        #
        # output
        # ------------------
        #   output_color : PIL Image paint segmentaion color (1024, 2048)
        #
        #
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear')
        output = interp(output).permute(0, 2, 3, 1)
        _, output = torch.max(output, -1)
        output = output.cpu().data[0].numpy().astype(np.uint8)
        output_color = colorize_mask(output)

        return output_color

    return output_to_image(predict_image)