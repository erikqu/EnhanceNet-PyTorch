# EnhanceNet-PyTorch
A PyTorch implementation of ENET-PA for Single Image Super Resolution (SISR).

![Screenshot](images/diagram.JPG)

Example from ENET paper

If you use this architecture in your work please cite the original paper:

```
@inproceedings{enhancenet,
  title={{EnhanceNet: Single Image Super-Resolution through Automated Texture Synthesis}},
  author={Sajjadi, Mehdi S. M. and Sch{\"o}lkopf, Bernhard and Hirsch, Michael},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  pages={4501--4510},
  year={2017},
  organization={IEEE},
  url={https://arxiv.org/abs/1612.07919/}
}
```

# Description 

ENET-PA here is implemented in PyTorch as there is no current implementation in PyTorch.  All credit goes to Sajjad et al. Adversarial learning along with perceptual loss (hence P+A).  The model is in the form of a GAN and does 4x upscaling of 64x64 images to 512x512.  

#### https://arxiv.org/abs/1612.07919/
