import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import cv2
# from scipy.misc import imread, imresize


class SYNYHIADataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        # todo:check mean
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        # self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
        #                       19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
        #                       26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        # Total 13 classes similar AdaptSegNet expreiment
        # self.id_to_trainid = {3: 0, 4: 1, 2: 2, 15: 6, 9: 7,
        #                       6: 8, 1: 10, 10:11, 17:12, 8:13,
        #                       19: 15, 12: 17, 11: 18}
        self.id_to_trainid = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7:5, 15: 6, 9: 7,
                              6: 8, 16: 9, 1: 10, 10: 11, 17: 12, 8: 13,
                              18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def convert(self, label):
        labels = {(0, 0, 0): 0, (70, 130, 180): 1, (70, 70, 70): 2, (128, 64, 128): 3, (244, 35, 232): 4,
                  (64, 64, 128): 5, (107, 142, 35): 6, (153, 153, 153): 7, (0, 0, 142): 8, (220, 220, 0): 9,
                  (220, 20, 60): 10, (119, 11, 32): 11, (0, 0, 230): 12, (250, 170, 160): 13, (128, 64, 64): 14,
                  (250, 170, 30): 15, (152, 251, 152): 16, (255, 0, 0): 17, (0, 0, 70): 18, (0, 60, 100): 19,
                  (0, 80, 100): 20, (102, 102, 156): 21, (102, 102, 156): 22}

        h, w, _ = label.shape
        new_label = np.zeros((h, w))
        # print(new_label.shape)
        for i in range(h):
            for j in range(w):
                try:
                    a = labels[(label[i, j][0], label[i, j][1], label[i, j][2])]
                except KeyError:
                    a = -100
                new_label[i, j] = a
        # print(np.unique(new_label))
        return new_label
    def color_code(self, image):
        labels = {(0, 0, 0): 0, (70, 130, 180): 1, (70, 70, 70): 2, (128, 64, 128): 3, (244, 35, 232): 4,
                  (64, 64, 128): 5, (107, 142, 35): 6, (153, 153, 153): 7, (0, 0, 142): 8, (220, 220, 0): 9,
                  (220, 20, 60): 10, (119, 11, 32): 11, (0, 0, 230): 12, (250, 170, 160): 13, (128, 64, 64): 14,
                  (250, 170, 30): 15, (152, 251, 152): 16, (255, 0, 0): 17, (0, 0, 70): 18, (0, 60, 100): 19,
                  (0, 80, 100): 20, (102, 102, 156): 21, (102, 102, 156): 22}
        labels = [list(k) for k, v in labels.items()]
        # h, w, _ = label.shape
        # new_label = np.zeros((h, w))
        # # print(new_label.shape)
        # for i in range(h):
        #     for j in range(w):
        #         try:
        #             a = labels[(label[i, j][0], label[i, j][1], label[i, j][2])]
        #         except KeyError:
        #             a = -100
        #         new_label[i, j] = a
        # print(np.unique(new_label))

        colour_codes = np.array(labels)
        x = colour_codes[image.astype(int)]
        print("color_code x =", x)
        return x

    def _scale(self, img, seg):
        h_s, w_s = 760, 1280
        # h_s, w_s = 720, 1312
        img_scale = img.resize((h_s, w_s), Image.BICUBIC)
        seg = (seg + 1).astype(np.uint8)
        seg_scale = cv2.resize(seg, (h_s, w_s), cv2.INTER_NEAREST)
        seg_scale = seg_scale.astype(np.int) - 1

        return img_scale, seg_scale

    def _crop(self, img, seg, cropSize=760, is_train=False):
        h_s, w_s = 760, 1280
        if is_train:
            # random crop
            x1 = random.randint(0, w_s - cropSize)
            y1 = random.randint(0, h_s - cropSize)
            img_crop = img[y1: y1 + cropSize, x1: x1 + cropSize, :]
            seg_crop = seg[y1: y1 + cropSize, x1: x1 + cropSize]
        else:
            # no crop
            img_crop = img
            seg_crop = seg

        return img_crop, seg_crop

    def _flip(self, img, seg):
        img_flip = img[:, ::-1, :].copy()
        seg_flip = seg[:, ::-1].copy()
        return img_flip, seg_flip

    def __getitem__(self, index):

        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        # label = Image.open(datafiles["label"])
        label = cv2.imread(datafiles["label"], cv2.IMREAD_UNCHANGED)[:, :, 2]
        # gt_lab = np.asarray(imageio.imread(gt_lab, format='PNG-FI'))
        name = datafiles["name"]


        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = cv2.resize(label, self.crop_size, cv2.INTER_NEAREST)
        # image, label = self._scale(image, label)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.int32)
        # random crop
        # image, label = self._crop(image, label, cropSize=760, is_train=self.is_mirror)
        # print("labels =", label)

        # image = np.asarray(image, np.float32)
        # label = np.asarray(label, np.int32)
        # label = self.convert(label)
        # label = self.color_code(label)
        #
        # print("float32 labels =", label)
        # print("float32 labels max=", np.max(label))
        # print("float32 labels shape=", label.shape)


        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        # print("label_copy =", label_copy)
        # print("float32 label_copy max=", np.max(label_copy))

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        # random flip
        # if self.is_mirror and random.choice([-1, 1]) > 0:
        #     # print("random flip")
        #     image, label_copy = self._flip(image, label_copy)

        return image.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    dst = SYNYHIADataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
