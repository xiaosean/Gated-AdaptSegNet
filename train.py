"""
Code concept modify from AdaptSegNet
    - Learning to Adapt Structured Output Space for Semantic Segmentation
    - https://arxiv.org/abs/1802.10349
    - https://github.com/wasidennis/AdaptSegNet

Code style modify Modify from MUNIT
    - https://github.com/NVlabs/MUNIT

Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse
import os
import time
from collections import OrderedDict


import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from trainer import AdaptSeg_Edge_Aux_Trainer
from trainer_spatial import AdaptSeg_Multi_Trainer
from util.util import get_all_data_loaders, get_config
from util.visualizer import Visualizer

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--config", type=str, default="./configs/default.yaml",
                        help="available options : DeepLab")
    return parser.parse_args()

def main():
    args = get_arguments()
    # cuda setting
    cudnn.enabled = True
    cudnn.benchmark = True
    # config setting
    CONFIG_PATH = args.config

    config = get_config(CONFIG_PATH)

    gpu = config["gpu"]
    # training setting
    num_steps = config['num_steps']

    # log setting
    snapshot_save_iter = config["snapshot_save_iter"]
    image_save_iter = config["image_save_iter"]

    log_dir = config["log_dir"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    snapshot_save_dir = config["snapshot_save_dir"]
    if not os.path.exists(snapshot_save_dir):
        os.makedirs(snapshot_save_dir)

    image_save_dir = config["image_save_dir"]
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    # data loader
    train_loader, target_loader = get_all_data_loaders(config)

    # model init
    print("config[d_model] =", config["d_model"])
    if config["d_model"] == "Spatail":  # ICIP version
        print("ICIP version")
        trainer = AdaptSeg_Multi_Trainer(config)
    else:  # ICME version
        print("ICME version")
        trainer = AdaptSeg_Edge_Aux_Trainer(config)

    print("config[restore] =", config["restore"])
    print("config[model]  =", config["model"])

    if config["restore_from"] == "None":
        print("# no pre-trained weight")
        print("# trained from scratch")
    elif config["restore"] and config["model"] != "DeepLabv3+":
        trainer.restore(model_name=config["model"], num_classes=config["num_classes"], restore_from=config["restore_from"])


    # Start training

    # set visualizer [tensorboard/html]
    if config["visualizer"]:
        visualizer = Visualizer(config)

    while True:
        for i_iter, (train_batch, target_batch) in enumerate(zip(train_loader, target_loader)):
            iter_start_time = time.time()
            torch.cuda.empty_cache()

            trainer.init_each_epoch(i_iter)
            trainer.update_learning_rate()
            # ====================== #
            #   Main training code   #
            # ====================== #

            # train G use source image
            src_images, labels, _, names = train_batch
            src_images = Variable(src_images).to(gpu)
            labels = Variable(labels).long().to(gpu)
            trainer.gen_source_update(src_images, labels, names)
            del src_images

            # train G use target image
            target_images, _, _, target_name = target_batch
            target_images = Variable(target_images).to(gpu)

            trainer.gen_target_update(target_images, target_name)
            del target_images

            # train discriminator use prior generator image
            trainer.dis_update()
            trainer.update_loss()

            # show log
            t = (time.time() - iter_start_time)
            if config["visualizer"]:
                visualizer.print_current_errors(num_steps, i_iter, trainer.loss_dict, t)
                visualizer.plot_current_errors(trainer.loss_dict, i_iter)
            else:
                trainer.show_each_loss()


            ### display output images
            if i_iter % image_save_iter == 0:
                # print("image_save_dir", image_save_dir)
                trainer.snapshot_image_save(dir_name=image_save_dir)
                if hasattr(trainer, 'snapshot_edge_save'):
                    trainer.snapshot_edge_save(dir_name=image_save_dir, labels=labels)

                if config["visualizer"]:
                    checkoutput_dir = config["image_save_dir"]
                    source_input_path = os.path.join(checkoutput_dir, "Image_source_domain_seg", '%s_input.png' % i_iter)
                    source_output_path = os.path.join(checkoutput_dir, "Image_source_domain_seg", '%s.png' % i_iter)
                    source_label_path = os.path.join(checkoutput_dir, "Image_source_domain_seg",
                                                     '%s_label.png' % i_iter)
                    if hasattr(trainer, 'snapshot_edge_save'):
                        source_edge_path = os.path.join(checkoutput_dir, "Image_source_domain_seg", '%s_edge.png' % i_iter)
                        target_edge_path = os.path.join(checkoutput_dir, "Image_target_domain_seg",
                                                        '%s_edge.png' % i_iter)

                    target_input_path = os.path.join(checkoutput_dir, "Image_target_domain_seg", '%s_input.png' % i_iter)
                    target_output_path = os.path.join(checkoutput_dir, "Image_target_domain_seg", '%s.png' % i_iter)
                    visuals = OrderedDict([('source_input', source_input_path),
                                           ('source_output', source_output_path),
                                           ('source_label', source_label_path),
                                           ('target_input', target_input_path),
                                           ('target_output', target_output_path),
                                           ])
                    if hasattr(trainer, 'snapshot_edge_save'):
                        visuals = OrderedDict([('source_input', source_input_path),
                                               ('source_output', source_output_path),
                                               ('source_label', source_label_path),
                                               ('source_edge', source_edge_path),
                                               ('target_input', target_input_path),
                                               ('target_output', target_output_path),
                                               ('target_edge', target_edge_path)
                                               ])
                    visualizer.display_current_results_by_path(visuals, i_iter, step=image_save_iter)

            # save final model .pth
            if i_iter == num_steps - 1:
                trainer.save_model(snapshot_save_dir=snapshot_save_dir)

            # save checkpoint .pth
            if i_iter % snapshot_save_iter == 0:
                trainer.save_model(snapshot_save_dir=snapshot_save_dir)
                torch.save(trainer.state_dict(),
                           os.path.join(snapshot_save_dir, 'GTA5_' + str(i_iter) + '_trainer_all.pth'))


if __name__ == "__main__":
    main()