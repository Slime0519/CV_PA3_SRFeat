# SRFeat

## Description
This repository contains my implementation RefSR method proposed in [SRFeat: Single Image Super-Resolution
with Feature Discrimination
](http://cg.postech.ac.kr/papers/2018_ECCV_SRFeat.pdf)

![](/Description_image/comparison_image.png)

### Generator
![](/Description_image/Generator.png)
    
Generator is also called SRResNet.    
It has 5 Residual blocks (simpler version), and reconstruct image by pixelshuffler(a.k.a subpixel convolutional layer)
 proposed in [ “Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network (CVPR 2016)](https://arxiv.org/pdf/1609.05158.pdf)
  

### Discriminator
![](/Description_image/Discriminator.png)
   
Discriminator has ordinary structure like traditional Discriminator of GAN.<br>
It has Sequential ( Conv + BN + LeakyReLU ) blocks and downsample input image alternately.   

### Perceptual loss
Proposed in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution ](https://arxiv.org/pdf/1603.08155) paper<br>
![](/Description_image/VGGLoss.png)<br>
To improve perceptual quality of result images, SRGAN use perceptual loss(a.k.a VGG loss).
This function compare two feature map extracted intermediate layer in pretrained VGG-19 network, we can easily implement this function. 
I used pytorch internal library and extract feature map easily.<br>

 original input | ![original input](/Description_image/original_input.png)  
 VGG22          | ![VGG22_1](/Description_image/VGG22_1.png){: width="224" height="224"} ![VGG22_2](/Description_image/VGG22_2.png){: width="224" height="224"} 
 VGG54          | ![VGG54_1](/Description_image/VGG54-1.png){: width="224" height="224"} ![VGG54_2](/Description_image/VGG54_2.png){: width="224" height="224"}

## Dataset
| Dataset name | usage               | link                                                                   |
|--------------|---------------------|------------------------------------------------------------------------|
| CUFED        | Training/Validation | https://drive.google.com/open?id=1hGHy36XcmSZ1LtARWmGL5OK1IUdWJi3I     |
| CUFED5       | Test                | https://drive.google.com/file/d/1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph/view |


## References
1. ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network", CVPR 2017](https://arxiv.org/pdf/1609.04802)
2. ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network", CVPR 2016](https://arxiv.org/pdf/1609.05158.pdf)
3. ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution", CVPR 2016 ](https://arxiv.org/pdf/1603.08155)