import os
import shutil

# from tensorboard_logger import configure, log_value
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

from model.deeplab_with_edge import Res_Deeplab as Res_Deeplab_Edge
from model.sp_discriminator import SP_FCDiscriminator
from model.gated_discriminator import Gated_Discriminator
from util.loss import CrossEntropy2d

class AdaptSeg_Edge_Aux_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(AdaptSeg_Edge_Aux_Trainer, self).__init__()
        self.hyperparameters = hyperparameters
        # input size setting
        self.input_size = (hyperparameters["input_size_h"], hyperparameters["input_size_w"])
        self.input_size_target = (hyperparameters["input_target_size_h"], hyperparameters["input_target_size_w"])

        # training setting
        self.num_steps = hyperparameters["num_steps"]

        # cuda setting
        self.gpu = hyperparameters['gpu']
        cudnn.benchmark = True

        # todo: remove this line without dev version
        assert hyperparameters["model"] == 'DeepLabEdge', True

        self.input_as_global_dis_features = hyperparameters['dis']['input_as_global_dis_features']
        self.input_as_foreground_dis_features = hyperparameters['dis']['input_as_foreground_dis_features']
        self.mask_as_foreground_dis_features = hyperparameters['dis']['mask_as_foreground_dis_features']

        # init G
        self.model = Res_Deeplab_Edge(num_classes=hyperparameters["num_classes"])

        # additional channel
        g_additional_channels = 0
        f_additional_channels = 0
        if self.input_as_global_dis_features:
            g_additional_channels += 3
        # set up Global Domain adpatation
        self.model_D = SP_FCDiscriminator(num_classes=hyperparameters['num_classes'] + g_additional_channels)

        if self.input_as_foreground_dis_features:
            f_additional_channels += 3
        if self.mask_as_foreground_dis_features:
            f_additional_channels += 1
        # set up Foreground Domain adpatation
        self.model_D_foreground = Gated_Discriminator(num_classes=hyperparameters['num_classes'] + f_additional_channels)

        self.model.train()
        self.model.cuda(self.gpu)
        self.model_D.train()
        self.model_D.cuda(self.gpu)
        self.model_D_foreground.train()
        self.model_D_foreground.cuda(self.gpu)

        # for dynamic adjust lr setting
        self.decay_power = hyperparameters['decay_power']

        # init optimizer
        self.lr_g = hyperparameters['lr_g']
        self.lr_d = hyperparameters['lr_d']

        self.momentum = hyperparameters['momentum']
        self.weight_decay = hyperparameters['weight_decay']
        self.beta1 = hyperparameters['beta1']
        self.beta2 = hyperparameters['beta2']
        self.D_beta1 = hyperparameters['D_beta1']
        self.D_beta2 = hyperparameters['D_beta2']
        self.init_opt()

        self.i_iter = 0
        self.source_label_path = None
        self.target_image_path = None

        # for generator
        self.lambda_seg = hyperparameters['gen']['lambda_seg']
        self.lambda_adv_target = hyperparameters['gen']['lambda_adv_target']
        self.lambda_adv_foreground = hyperparameters['gen']['lambda_adv_foreground']
        self.lambda_adv_edge = hyperparameters['gen']['lambda_adv_edge']
        self.decay_power = hyperparameters['decay_power']

        # for discriminator
        self.adv_loss_opt = hyperparameters['dis']['adv_loss_opt']

        self.source_image = None
        self.target_image = None
        self.pred_real_edge = None
        self.pred_fake_edge = None

        # for [log / check output]
        self.loss_names = ['Seg', 'Edge_Seg', 'Global_GAN_dis', 'Global_GAN_adv', 'Foreground_GAN_dis', 'Foreground_GAN_adv']
        self.loss_dict = {k: 0 for k in self.loss_names}

        # resize to source size
        self.interp = nn.Upsample(size=self.input_size, align_corners=True, mode='bilinear')

        # resize to target size
        self.interp_target = nn.Upsample(size=self.input_size_target, align_corners=True,
                                         mode='bilinear')

    def init_opt(self):
        self.optimizer_G = optim.SGD([p for p in self.model.parameters() if p.requires_grad],
                                     lr=self.lr_g, momentum=self.momentum, weight_decay=self.weight_decay)
        self.optimizer_G.zero_grad()
        self._adjust_learning_rate_G(self.optimizer_G, 0)

        self.optimizer_D = optim.Adam([p for p in self.model_D.parameters() if p.requires_grad],
                                      lr=self.lr_d, betas=(self.beta1, self.beta2))
        self.optimizer_D.zero_grad()

        self.optimizer_D_foreground = optim.Adam([p for p in self.model_D_foreground.parameters() if p.requires_grad],
                                                 lr=self.lr_d, betas=(self.D_beta1, self.D_beta2))
        self.optimizer_D_foreground.zero_grad()

    def forward(self, images):
        self.eval()
        predict_seg, _ = self.model(images)
        self.train()
        return predict_seg


    def gen_source_update(self, images, labels, label_path=None):
        """
                Input source domain image and compute segmentation loss.

                :param images:
                :param labels:
                :param label_path: just for save path to record model predict, use in  snapshot_image_save function

                :return:
                """
        self.optimizer_G.zero_grad()

        # Disable D backpropgation, we only train G
        for param in self.model_D.parameters():
            param.requires_grad = False

        for param in self.model_D_foreground.parameters():
            param.requires_grad = False

        self.source_label_path = label_path

        # get predict output
        pred_source_real, pred_source_edge = self.model(images)

        # resize to source size
        pred_source_real = self.interp(pred_source_real)
        pred_source_edge = self.interp(pred_source_edge)

        # in source domain compute segmentation loss
        seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
        seg_loss = seg_loss(pred_source_real, labels)


        bce_loss = nn.BCEWithLogitsLoss()
        attn_loss = bce_loss(pred_source_edge,
                             self.get_foreground_attention(
                                 labels.view(labels.shape[0], -1, labels.shape[1], labels.shape[2]).cuda()))

        # proper normalization
        seg_loss = self.lambda_seg*seg_loss + self.lambda_adv_edge*attn_loss
        seg_loss.backward()

        # save image for discriminator use
        self.source_image = pred_source_real.detach()
        self.pred_real_edge = nn.Sigmoid()(pred_source_edge).detach()
        self.source_input_image = images.detach()

        # record log
        self.loss_dict['Seg'] += seg_loss.item()
        self.loss_dict['Edge_Seg'] += self.lambda_adv_edge * attn_loss.item()
        del attn_loss, seg_loss


    def gen_target_update(self, images, image_path):
        """
                 Input target domain image and compute adversarial loss.

                :param images:
                :param image_path: just for save path to record model predict, use in  snapshot_image_save function
                :return:
                """

        self.target_image_path = image_path
        # get predict output
        pred_target_fake, pred_target_edge = self.model(images)
        # resize to target size
        pred_target_fake = self.interp_target(pred_target_fake)
        pred_target_edge = self.interp_target(pred_target_edge)

        net_input = F.softmax(pred_target_fake, dim=1)
        pred_fake_edge = nn.Sigmoid()(pred_target_edge)

        if self.input_as_global_dis_features:
            d_out_fake = self.model_D(torch.cat((net_input, images), dim=1))
        else:
            d_out_fake = self.model_D(net_input)

        if self.input_as_foreground_dis_features:
            d_out_foreground_fake = self.model_D_foreground(torch.cat((net_input, images), dim=1), label=pred_fake_edge)
        else:
            d_out_foreground_fake = self.model_D_foreground(net_input, label=pred_fake_edge)

        if self.adv_loss_opt == "hinge":
            loss_adv = self.loss_hinge_gen(d_out_fake)
            loss_adv_foreground = self.loss_hinge_gen(d_out_foreground_fake)
        else:
            loss_adv = self._compute_adv_loss_real(d_out_fake)
            loss_adv_foreground = self._compute_adv_loss_real(d_out_foreground_fake)

        loss = loss_adv + self.lambda_adv_foreground * loss_adv_foreground
        loss = self.lambda_adv_target * loss
        loss.backward()

        # save image for discriminator use
        self.target_image = pred_target_fake.detach()
        self.target_input_image = images.detach()
        self.pred_fake_edge = pred_fake_edge.detach()

        self.loss_dict['Global_GAN_adv'] += loss_adv.item()
        self.loss_dict['Foreground_GAN_adv'] += loss_adv_foreground.data.item()
        del loss, loss_adv, loss_adv_foreground

    def dis_update(self):
        """
                use [gen_source_update / gen_target_update]'s image to discriminator,
                so you  don' t need to give any parameter
                """
        self.optimizer_D.zero_grad()
        self.optimizer_D_foreground.zero_grad()

        # Enable D backpropgation, train D
        for param in self.model_D.parameters():
            param.requires_grad = True

        for param in self.model_D_foreground.parameters():
            param.requires_grad = True

        # we don't train target's G weight, we only train source'G
        self.source_image = self.source_image.detach()
        self.pred_real_edge = self.pred_real_edge.detach()
        self.target_image = self.target_image.detach()
        self.pred_fake_edge = self.pred_fake_edge.detach()

        # compute adv loss function

        net_input = F.softmax(self.source_image, dim=1)
        if self.input_as_global_dis_features:
            d_out_real = self.model_D(torch.cat((net_input, self.source_input_image), dim=1))
        else:
            d_out_real = self.model_D(net_input)

        if self.input_as_foreground_dis_features:
            d_out_foreground_real = self.model_D_foreground(torch.cat((net_input, self.source_input_image), dim=1),
                                                               label=self.pred_real_edge)
        else:
            d_out_foreground_real = self.model_D_foreground(net_input, label=self.pred_real_edge)

        net_input = F.softmax(self.target_image, dim=1)

        if self.input_as_global_dis_features:
            d_out_fake = self.model_D(torch.cat((net_input, self.target_input_image), dim=1))
        else:
            d_out_fake = self.model_D(net_input)

        # cobime predict and use predict output get edge
        if self.input_as_foreground_dis_features:
            d_out_foreground_fake = self.model_D_foreground(torch.cat((net_input, self.target_input_image), dim=1), label=self.pred_fake_edge)
        else:
            d_out_foreground_fake = self.model_D_foreground(net_input, label=self.pred_fake_edge)


        # foreground part
        if self.adv_loss_opt == "hinge":
            loss = self.loss_hinge_dis(d_out_foreground_fake, d_out_foreground_real)
        else:
            loss_fake = self._compute_adv_loss_fake(d_out_foreground_fake, self.adv_loss_opt)
            loss_real = self._compute_adv_loss_real(d_out_foreground_real, self.adv_loss_opt)
            loss = (loss_real + loss_fake)/2
            del loss_fake, loss_real
        loss.backward()
        self.loss_dict['Foreground_GAN_dis'] += loss.data.item()

        # global part
        if self.adv_loss_opt == "hinge":
            loss = self.loss_hinge_dis(d_out_fake, d_out_real)
        else:
            loss_fake = self._compute_adv_loss_fake(d_out_fake, self.adv_loss_opt)
            loss_real = self._compute_adv_loss_real(d_out_real, self.adv_loss_opt)
            loss = (loss_real + loss_fake)/2
            del loss_fake, loss_real
        loss.backward()
        self.loss_dict['Global_GAN_dis'] += loss.data.item()

        del loss

    def update_loss(self):
        self.optimizer_G.step()
        self.optimizer_D.step()
        self.optimizer_D_foreground.step()

    def loss_hinge_dis(self, dis_fake, dis_real):
        loss = torch.mean(F.relu(1. - dis_real))
        loss += torch.mean(F.relu(1. + dis_fake))
        return loss

    def loss_hinge_gen(self, g_fake):
        loss = -torch.mean(g_fake)
        return loss

    def show_each_loss(self):
        message = '(epoch: %d, iters: %d) ' % (self.num_steps, self.i_iter)
        for k, v in self.loss_dict.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def channel_to_label(self, pred):
        pred = pred.permute(0, 2, 3, 1)
        _, output = torch.max(pred, -1)
        return output

    def get_foreground_attention(self, label):
        foreground_attn = torch.cuda.ByteTensor(label.size()).zero_()
        foreground_map = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
        for value in foreground_map:
            foreground_attn[label == value] = 1
        self.saliency_mask = foreground_attn.detach().float()

        return foreground_attn.float()


    def _compute_adv_loss_real(self, d_out_real, loss_opt="bce", label=0):
        """
                compute adversarial loss function, can choose loss opt
                :param d_out_fake:
                :param loss_opt:  [wgan-gp / hinge / bce]
                :param label:
                :return:
                """
        assert loss_opt == "bce" or loss_opt == "wgan-gp"
        d_loss_real = None
        if loss_opt == 'wgan-gp':
            d_loss_real = - d_out_real.mean()
        elif loss_opt == 'bce':
            bce_loss = torch.nn.BCEWithLogitsLoss()
            d_loss_real = bce_loss(d_out_real,
                                   Variable(torch.FloatTensor(d_out_real.data.size()).fill_(label)).cuda(
                                       self.gpu))
        return d_loss_real

    def _compute_adv_loss_fake(self, d_out_fake, loss_opt="bce", label=1):
        """
                compute adversarial loss function, can choose loss opt
                :param d_out_fake:
                :param loss_opt:  [wgan-gp / hinge / bce]
                :param label:
                :return:
                """
        d_loss_fake = None
        if loss_opt == 'wgan-gp':
            d_loss_fake = - d_out_fake.mean()
        elif loss_opt == 'bce':
            bce_loss = torch.nn.BCEWithLogitsLoss()
            d_loss_fake = bce_loss(d_out_fake,
                                   Variable(torch.FloatTensor(d_out_fake.data.size()).fill_(label)).cuda(
                                       self.gpu))
        return d_loss_fake

    def _compute_seg_loss(self, pred, label):
        """
                This function returns cross entropy loss for semantic segmentation
                """
        # out shape batch_size x channels x h x w -> batch_size x channels x h x w
        # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
        label = Variable(label.long()).cuda(self.gpu)
        criterion = CrossEntropy2d().cuda(self.gpu)

        return criterion(pred, label)

    def _lr_poly(self, base_lr, i_iter, max_iter, power):
        return base_lr * ((1 - float(i_iter) / max_iter) ** power)

    def _adjust_learning_rate_D(self, optimizer, i_iter):
        lr = self._lr_poly(self.lr_d, i_iter, self.num_steps, self.decay_power)
        for i, group in enumerate(optimizer.param_groups):
            optimizer.param_groups[i]['lr'] = lr

    def _adjust_learning_rate_G(self, optimizer, i_iter):
        lr = self._lr_poly(self.lr_g, i_iter, self.num_steps, self.decay_power)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def init_each_epoch(self, i_iter):
        self.i_iter = i_iter
        self.loss_dict = {k: 0 for k in self.loss_names}


    def update_learning_rate(self):
        if self.optimizer_G:
            self.optimizer_G.zero_grad()
            self._adjust_learning_rate_G(self.optimizer_G, self.i_iter)

        if self.optimizer_D:
            self.optimizer_D.zero_grad()

        if self.optimizer_D_foreground:
            self.optimizer_D_foreground.zero_grad()

    def snapshot_image_save(self, dir_name="check_output/", src_save=True, target_save=True):
        """
                check model training status,
                will output image to config["image_save_dir"]
                """
        if not os.path.exists(os.path.join(dir_name, "Image_source_domain_seg")):
            os.makedirs(os.path.join(dir_name, "Image_source_domain_seg"))
        if not os.path.exists(os.path.join(dir_name, "Image_target_domain_seg")):
            os.makedirs(os.path.join(dir_name, "Image_target_domain_seg"))

        if src_save:
            # save image
            dataset_name = self.hyperparameters["source_dataset"]
            input_image_name = os.path.join("data", dataset_name, "images", self.source_label_path[0])
            save_name = os.path.join(dir_name, "Image_source_domain_seg", '%s_input.png' % self.i_iter)
            shutil.copyfile(input_image_name, save_name)

            # save label
            if dataset_name == "SYNTHIA":
                label_name = os.path.join("data", dataset_name, "labels-rgb", self.source_label_path[0])
            else:
                label_name = os.path.join("data", dataset_name, "labels", self.source_label_path[0])
            save_name = os.path.join(dir_name, "Image_source_domain_seg", '%s_label.png' % self.i_iter)
            shutil.copyfile(label_name, save_name)

            # save output image
            paint_predict_image(self.source_image).save('check_output/Image_source_domain_seg/%s.png' % self.i_iter)

        if target_save:
            target_name = os.path.join("data", "Cityscapes", "data", "leftImg8bit", "train", self.target_image_path[0])
            save_name = os.path.join(dir_name, "Image_target_domain_seg", '%s_input.png' % self.i_iter)
            shutil.copyfile(target_name, save_name)
            paint_predict_image(self.target_image).save('check_output/Image_target_domain_seg/%s.png' % self.i_iter)

    def snapshot_edge_save(self, dir_name="check_output/", src_save=True, target_save=True, labels=None):
        """
                check model training status,
                will output image to config["image_save_dir"]
                """
        if not os.path.exists(os.path.join(dir_name, "Image_source_domain_seg")):
            os.makedirs(os.path.join(dir_name, "Image_source_domain_seg"))
        if not os.path.exists(os.path.join(dir_name, "Image_target_domain_seg")):
            os.makedirs(os.path.join(dir_name, "Image_target_domain_seg"))

        # save output image
        if src_save:
            # record edge attn
            self.tensor_to_PIL(self.saliency_mask).save(
                'check_output/Image_source_domain_seg/%s_compute_attn.png' % self.i_iter)

            self.tensor_to_PIL(self.pred_real_edge).save(
                'check_output/Image_source_domain_seg/%s_edge.png' % self.i_iter)

        if target_save:
            self.tensor_to_PIL(self.pred_fake_edge).save(
                'check_output/Image_target_domain_seg/%s_edge.png' % self.i_iter)


    def save_model(self, snapshot_save_dir="./model_save"):
        """
                save model to .pth file
                will output model to config["snapshot_save_dir"]
                """
        print('taking pth in shapshot dir ...')
        torch.save(self.model.state_dict(), os.path.join(snapshot_save_dir, 'GTA5_' + str(self.i_iter) + '.pth'))
        torch.save(self.model_D.state_dict(), os.path.join(snapshot_save_dir, 'GTA5_' + str(self.i_iter) + '_D1.pth'))
        torch.save(self.model_D_foreground.state_dict(),
                   os.path.join(snapshot_save_dir, 'GTA5_' + str(self.i_iter) + '_D_foreground.pth'))

    def restore(self, model_name=None, num_classes=19, restore_from=None):

        # self.model = Res_Deeplab(num_classes=num_classes)
        print("check restore from", restore_from)
        if restore_from[:4] == 'http':
            saved_state_dict = model_zoo.load_url(restore_from)
            new_params = self.model.state_dict().copy()
            for i in saved_state_dict:
                # Scale.layer5.conv2d_list.3.weight
                i_parts = i.split('.')
                # if not num_classes == 19 or not i_parts[1] == 'layer5':
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            self.model.load_state_dict(new_params)
            del new_params
        else:
            saved_state_dict = torch.load(restore_from)
            self.model.load_state_dict(saved_state_dict)
        self.init_opt()

    # todo: move to util.py
    def tensor_to_PIL(self, tensor):
        pilTrans = transforms.ToPILImage()
        image = pilTrans(tensor.cpu().squeeze(0))
        return image


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
        interp = nn.Upsample(size=(1024, 2048), align_corners=True, mode='bilinear')
        output = interp(output).permute(0, 2, 3, 1)
        _, output = torch.max(output, -1)
        output = output.cpu().data[0].numpy().astype(np.uint8)
        output_color = colorize_mask(output)

        return output_color

    return output_to_image(predict_image)
