# Domain Adapt for Semantic Segmentation

## Code will be avaliable in July. Currently, I am working on the journal version.

Pytorch implementation domain adaption of semantic segmentation from the synthetic dataset (source domain) to the real dataset (target domain). Based on [AdaptSegNet].

Contact: Yong-Xiang Lin (xiaosean5408 at gmail dot com)

## Paper

Please cite our paper if you find it is useful for your research.

Yong-Xiang Lin, Daniel Stanley Tan, Wen-Huang Cheng, Kai-Lung Hua. “Adapting Semantic Segmentation of Urban Scenes via Mask-aware Gated Discriminator,” In Proceedings of the IEEE International Conference on Multimedia & Expo (ICME), Shanghai, China, Jul. 8 - 12, 2019. (Oral)

Yong-Xiang Lin, Daniel Tan, Wen-Huang Cheng, Yung-Yao Chen, Kai-Lung Hua. “Spatially-aware Domain Adaptation for Semantic Segmentatino of Urban Scenes,” In Proceedings of the IEEE International Conference on Image Processing (ICIP), Taipei, Taiwan, September 22-25, 2019. (Oral)

## Example Results

![figure/Output.png]

## Model Architecture

![figure/Overview.png]

## Prerequisites
- Python 3
- NVIDIA GPU (10G up, I used Nvidia GTX 1080Ti) + CUDA cuDNN
- Pytorch(Vesion needs 0.4.1 up because I use spectral_norm)

## Getting Started
### Installation

#### You can choose [pip install / conda install by yml]()
* #### Intall method 1. use pip 
    - Install PyTorch and dependencies from http://pytorch.org
    ```bash
    pip install dominate, scipy, matplotlib, pillow, pyyaml
    ```
* ####  Intall method 2. use conda env 
    ```bash
    conda env create -f environment.yml
    ```
    - On Windows
        ```bash
        activate Gated-AdaptSeg 
        ```
    - On Linux
        ```bash
        conda activate Gated-AdaptSeg 
        ```
    ```bash
    pip install dominate
    ```

- Clone this repo:
```bash
git clone https://github.com/xiaosean/Gated-AdaptSeg
cd Gated-AdaptSeg
```

## Dataset
* Download the [GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/) as the source domain, and put it in the `data/GTA5` folder

* Download the [Cityscapes Dataset](https://www.cityscapes-dataset.com/) as the target domain, and put it in the `data/Cityscapes` folder

## Testing
* Download the Gated pre-trained model [link](https://drive.google.com/file/d/1Sft6duJcgciJ2fR0oQMzf9eUt5Tinjz9/view)

Note: This version classification only use ASPP Module [6, 12]，Similar as [AdaptSegNet]

```
python evaluate_cityscapes.py --restore-from ./Gated-GTA5-Cityscapes_250000.pth
```

* Compute the IoU on Cityscapes (thanks to the code from [VisDA Challenge](http://ai.bu.edu/visda-2017/))
```
python compute_iou.py ./data/Cityscapes/data/gtFine/val result/cityscapes
```

## Training Examples

Firstly, Using [FastPhotoStyle], transfer GTA5 dataset to Cityscapes photo style.

![figure/FastphotoDemo.png]

Second, 
* Train the GTA5-to-Cityscapes model - 
    * Config use => configs/edge_TTUR_v1.yaml
```
python train.py
```

## Visulization
Open ./check_output/index.html

## Support Tensorboard
```
tensorboard --logdir=check_output\log --host=127.0.0.1
```

## Model 

Using model/gated_discriminator
> Yong-Xiang Lin, Daniel Stanley Tan, Wen-Huang Cheng, Kai-Lung Hua. “Adapting Semantic Segmentation of Urban Scenes via Mask-aware Gated Discriminator,” In Proceedings of the IEEE International Conference on Multimedia & Expo (ICME), Shanghai, China, Jul. 8 - 12, 2019. (Oral)

Using model/sp_cgan_discriminator and download the spatial prior from GTA5.

Spatial prior download go to ECCV'18 [cbst] download the Spatial prior:[Spatial prior brrowed from cbst]

> Yong-Xiang Lin, Daniel Tan, Wen-Huang Cheng, Yung-Yao Chen, Kai-Lung Hua. “Spatially-aware Domain Adaptation for Semantic Segmentatino of Urban Scenes,” In Proceedings of the IEEE International Conference on Image Processing (ICIP), Taipei, Taiwan, September 22-25, 2019. (Oral)


## Acknowledgment
This code is heavily borrowed from [AdaptSegNet].
Visualize part is heavily borrowed from [pix2pixHD].
Spatial prior used [cbst].

Especially, thanks these paper and code,
[Learning to Adapt Structured Output Space for Semantic Segmentation](https://arxiv.org/abs/1802.10349) <br/>
[Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/home)\*, [Wei-Chih Hung](https://hfslyc.github.io/)\*, [Samuel Schulter](https://samschulter.github.io/), [Kihyuk Sohn](https://sites.google.com/site/kihyuksml/), [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/index.html) and [Manmohan Chandraker](http://cseweb.ucsd.edu/~mkchandraker/) <br/>
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018 (**spotlight**) (\* indicates equal contribution).

[Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training](http://openaccess.thecvf.com/content_ECCV_2018/html/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.html) <br/>
Yang Zou, Zhiding Yu, B.V.K. Vijaya Kumar, Jinsong Wangl; The European Conference on Computer Vision (ECCV), 2018, pp. 289-305 <br/>
In ECCV 2018 paper <br/>


## Note
The model and code are available for non-commercial research purposes only.
* May, 05, 2019: Add readme.txt

[AdaptSegNet]:https://github.com/wasidennis/AdaptSegNet
[FastPhotoStyle]:https://github.com/NVIDIA/FastPhotoStyle
[pix2pixHD]:https://github.com/NVIDIA/pix2pixHD
[cbst]:https://github.com/yzou2/cbst
[Spatial prior brrowed from cbst]:https://www.dropbox.com/s/o6xac8r3z30huxs/prior_array.mat?dl=0
