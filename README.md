# GTNet: Generative Transfer Network for Zero-Shot Object Detection (AAAI 2020)

Official code for AAAI 2020 paper [GTNet: Generative Transfer Network for Zero-Shot Object Detection](https://arxiv.org/abs/2001.06812). 

<p align="center">
	<img src=image/gtnet.png width=80% />
<p align="center">

## Introduction

We propose a Generative Transfer Network (GTNet) for zero-shot object detection (ZSD). GTNet consists of an Object Detection Module and a Knowledge Transfer Module. The Object Detection Module can learn large-scale seen domain knowledge. The Knowledge Transfer Module leverages a feature synthesizer to generate unseen class features, which are applied to train a new classification layer for the Object Detection Module. In order to synthesize features for each unseen class with both the intra-class variance and the IoU variance, we design an IoU-Aware Generative Adversarial Network (IoUGAN) as the feature synthesizer, which can be easily integrated into GTNet. Specifically, IoUGAN consists of three unit models: Class Feature Generating Unit (CFU), Foreground Feature Generating Unit (FFU), and Background Feature Generating Unit (BFU). CFU generates unseen features with the intra-class variance conditioned on the class semantic embeddings. FFU and BFU add the IoU variance to the results of CFU, yielding class-specific foreground and background features, respectively. We evaluate our method on three public datasets and the results demonstrate that our method performs favorably against the state-of-the-art ZSD approaches.


## Dependencies
1. PyTorch 0.3.1
2. Python 3.5

## Detection Results on MSCOCO
<p align="center">
	<img src=image/detection_results.png width=80% />
<p align="center">


## Citation

If you find this code useful in your research, please consider citing:
```
@inproceedings{zhao2020gtnet,
  title={GTNet: Generative Transfer Network for Zero-Shot Object Detection},
  author={Shizhen, Zhao and Changxin, Gao and Yuanjie, Shao and Lerenhan, Li and Changqian, Yu and Zhong, Ji and Nong, Sang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2020}
}
```

## Acknowledgement

Our code is largely based on [Faster RCNN](https://github.com/jwyang/faster-rcnn.pytorch), and we thank the authors for their implementation. Please also consider citing their wonderful code base. 

```
@misc{Ren_2015,
    title={{Faster r-cnn: Towards real-time object detection with region proposal networks},
    author={Ren, Shaoqing, Kaiming He, Ross Girshick, and Jian Sun},
    booktitle={Advances in neural information processing systems},
    year={2015}
}
```

## Contact
If you have any questions, you can email me (xbrainlab@gmail.com). 
