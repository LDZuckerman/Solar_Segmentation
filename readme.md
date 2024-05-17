NOTE: this is an ongoing project and is still under development.

## Motivation

Classical methods of segmenting features within the photosphere rely on simple, empirically-driven, rule-based algorithms. These can provide wildly inconsistent results, and are highly sensitive to variability in image resolution and quality. What's more, they do not incorporate the physics underlying photospheric structure, and thus cannot in an of themselves tell us anything meaningfull about these features. 

Unsupervised models learn their own definitions of features. Using an unsupervised (or semi-supervised) approach, I explore the properties intrinsic to these features, and aim to create a more nuance and scientifically interesting framework for photospheric feature segmentation. 

## Approach

While my methods remain unsupervised (training is performed without labeled segmentations), I do search for models that predict feature classes similar to those identified by solar physics experts. I iteratively train models using modified arhcitecture, parameters, and data, and in doing so learn what information is most important in defining these features. For quick visual evaluation, I use the predictions of a simple segmentation algorithm that produces results deemed "correct" by solar physists.  

Along with other models, I implement a **WNet** [[1]](#1) architecture which trains based on ``reconstructions" of input images. As a convolutional nueral network, the WNet allows spatial information to be preserved during training, which is important for an image segmentation task.  

![alt text](https://github.com/LDZuckerman/Solar_Segmentation/blob/master/WNet.png)
*Schematic of the WNet architecture (image from [[1]](#1))*

## References

<a id="1">[1]</a> 
Xide, X. (2017). 
Xia, X., & Kulis, B. (2017). W-Net: A Deep Model for Fully Unsupervised Image Segmentation. ArXiv, abs/1711.08506.