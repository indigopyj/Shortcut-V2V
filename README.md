# Shortcut-V2V: Compression Framework for Video-to-Video Translation based on Temporal Redundancy Reduction (ICCV 2023)

### [Project page](https://shortcut-v2v.github.io/) | [Paper](https://arxiv.org/abs/2308.08011)

Chaeyeon Chung*, Yeojeong Park*, Seunghwan Choi, Munkhsoyol Ganbat, and Jaegul Choo

\* indicates equal contributions

# Abstract

> Video-to-video translation aims to generate video frames of a target domain from an input video. Despite its usefulness, the existing video-to-video translation methods require enormous computations, necessitating their model compression for wide use. While there exist compression methods that improve computational efficiency in various image/video tasks, a generally-applicable compression method for video-to-video translation has not been studied much. In response, this paper presents *Shortcut-V2V*, a general-purpose compression framework for video-to-video translation. Shortcut-V2V avoids full inference for every neighboring video frame by approximating the intermediate features of a current frame from those of the preceding frame. Moreover, in our framework, a newly-proposed block called AdaBD adaptively blends and deforms features of neighboring frames, which makes more accurate predictions of the intermediate features possible. We conduct quantitative and qualitative evaluations using well-known video-to-video translation models on various tasks to demonstrate the general applicability of our framework. The results show that Shoutcut-V2V achieves comparable performance compared to the original video-to-video translation model while saving **3.2-5.7x** computational cost and **7.8-44x** memory at test time.
> 

# Prerequisites

---

- Linux
- Python 3.8
- torch == 1.9.1 , torchvision == 0.10.0
- Please follow the environment setting from the original repository.
    - [Unsupervised RecycleGAN](https://github.com/wangkaihong/Unsup_Recycle_GAN)
    - [VToonify](https://github.com/williamyang1991/VToonify)
- Extra dependencies for Shortcut-V2V:

```bash
pip install -r requirements.txt
```

# Dataset

---

- Viper-to-Cityscapes
    - Viper dataset can be downloaded [here](https://playing-for-benchmarks.org/download/). We used ‘jpg’ image dataset due to the limited space. We split the validation set from the train set following [the previous work](https://github.com/aayushbansal/Recycle-GAN/).
    - Cityscapes dataset is available [here](https://www.cityscapes-dataset.com/downloads/). Note that we need to deal with video data, so we downloaded “[leftImg8bit_sequence_trainvaltest.zip (324GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=14)”. Don’t be confused by downloading image dataset.
- Label-to-Viper / Viper-to-Label
    - Both image and label datasets are originally provided by [RecycleGAN](https://www.dropbox.com/s/qhu29y5cx4lyfun/Viper_data.tar.gz?dl=0)(ECCV 2018). They have different resolution (256x256) with the dataset mentioned above.
- Dataset Structure

```jsx
$dataset_root
|--Viper  
| |--train
|	| |--img
| | |--cls
| |--val
| | |--img
| | |--cls                  
|--Cityscapes_sequence
| |--leftImg8bit
| | |--train
| | |--val                      
```

# Inference

---

| Model | Dataset | Type | Link |
| --- | --- | --- | --- |
| Unsupervised RecycleGAN | Viper-to-Cityscapes | Teacher | [here](https://drive.google.com/file/d/1ZZDL44Xh0uoBKIQOq59vN1hCoyuANfU1/view?usp=sharing) |
| Unsupervised RecycleGAN | Viper-to-Cityscapes | Shortcut-V2V | [here](https://drive.google.com/file/d/12L9QPuEkuXNlL51PbNiARwuL6Wprueh0/view?usp=sharing) |
| VToonify | - | Teacher |  |
| VToonify | - | Shortcut-V2V |  |

### Unsupervised RecycleGAN(AAAI 2022) + Shortcut-V2V

**Viper-to-Cityscapes**

```bash
bash scripts/unsup_recycle/test_v2c.sh
```


---

There are two steps for training. First, train an original generator following the original repository. Then, train our method, Shortcut-V2V. If you already have pretrained checkpoints of the original generator, you can skip the first step and easily attach our method.

### Unsupervised RecycleGAN(AAAI 2022) + Shortcut-V2V

**Viper-to-Cityscapes**

```bash
bash scripts/unsup_recycle/train_v2c.sh
```


# TODO
- [x] Unsupervised RecycleGAN + Shortcut-V2V (Viper-to-Cityscapes)
- [ ] VToonify + Shortcut-V2V