Team Members details :
1. Ramasish Parida (522232) (Leader)
2. Kannikanti Pardha Saradhi (522145)
3. Abhishek Singh (522101)
4. Vipul Yadav (522255)
5. Rekulapally Sai Kumar (522234)
6. Muddisetti Gnaneswara (522208)

Attention-based U-Net for Land Cover Segmentation and Classification

This repository contains an implementation of an Attention-based U-Net model designed for semantic segmentation of various landforms from satellite imagery.

Overview:

The `LightAttentionUNet` model processes satellite images (3-channel RGB) and outputs a 7-class segmentation map corresponding to different land cover types:
1.Urban land
2.Agriculture land
3.Rangeland
4.Forest land
5.Water
6.Barren land
7.Unknown/background areas

Model Architecture:

Downsampling Path (Encoder):

The encoder follows a progressively downsampling architecture:

1. Initial Feature Extraction: The input satellite image (3 channels) is processed through the first convolutional block, producing 32 feature maps while maintaining spatial dimensions.

2. Progressive Downsampling: The model applies a series of operations:
   Max pooling reduces the spatial dimensions by half
   Convolutional blocks extract increasingly abstract features
   The number of feature channels increases: 32 → 64 → 128 → 256 → 512

3. Convolutional Blocks: Each block consists of two 3×3 convolutions with batch normalization and ReLU activations.

The model uses gradient checkpointing for memory efficiency during training.

Attention Mechanism:

The attention gates are designed to highlight relevant features and suppress irrelevant ones:

1. Gate Structure: Each attention gate takes two inputs:
   Features from the decoder path
   Skip connection features from the encoder

2. Attention Computation: The model computes attention coefficients that weight the importance of different spatial locations in the feature maps.

3. Attention Focus: The attention gates learn to focus on specific regions in the feature maps that are most relevant for land cover classification.

 Upsampling Path (Decoder):

The decoder gradually restores spatial resolution while incorporating contextual information:

1. Transposed Convolutions: Each upsampling step uses transposed convolutions to double the spatial dimensions while reducing channel depth.

2. Skip Connections with Attention: Before concatenating features from the encoder path, the model applies attention gates to focus on relevant features.

3. Progressive Up-sampling: The process continues through four up-sampling stages:
   512 → 256 → 128 → 64 → 32 channels
   1/16 → 1/8 → 1/4 → 1/2 → original resolution

Final Segmentation Output:

The final layer applies a 1×1 convolution to map the 32-channel feature space to 7 output channels. During inference, an argmax operation is applied to predict the most likely class for each pixel.

Loss Function and Training:

The model is trained using a combination of Cross-Entropy and Dice Loss, with class weights to handle class imbalance common in satellite imagery.

The attention-based approach provides significant advantages over standard U-Net for land cover segmentation, as it helps the model to focus on relevant features while ignoring background noise.

LINK TO VIDEO PRESENTATION : https://drive.google.com/file/d/10SPEkFdPNcF0JD2ijXnvwQfba3oq5SHX/view?usp=sharing

