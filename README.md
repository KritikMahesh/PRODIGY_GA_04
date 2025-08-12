# 🎨 Image-to-Image Translation with Conditional GAN (cGAN)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/kritikmahesh/image-to-image-translation-with-cgan.55a8bb1b-5cf8-4aa0-90d3-ca7b04121a8d.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20250812/auto/storage/goog4_request%26X-Goog-Date%3D20250812T082401Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D888b3c9680f18fd206f8e0f3bc220e02f133f1ce503c02fa2ef2bae68a2c59b8a1ce164d6e4c61b5e57eff770d1fd96f0e7a127cbf1aba10b09d4436cdc6d1e6c40a15c07a6a2ef5fe6f1745b702059e7239f02056c8ff76f18f1f343b3bea144b5bf6330fb60efeff09cc7bb23ed316dff1a7863fc6ae1929a7446ebc62491d5ae5adc3d948ea286cc3ae84831ebd107da3276644678dc54b1c0b38ae899db855c2a571375536d2e58bd581926344187410705304a1c288561a678f89dca5710eaa5611e3d4ba842f6b06d030643f8e5ccb996ab1f4800622c9aeb38e1a057dc6d3eab3fd817b1d6934220b9eb4c00125418cc55c09f012a80c30a9981cdcd0)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)

> **Grayscale to Color Image Translation using Conditional GANs**

This repository contains an implementation of **Image-to-Image Translation** using **Conditional Generative Adversarial Networks (cGANs)** that converts grayscale images to colorized versions. The project uses the **TensorFlow Flowers dataset** and demonstrates how cGANs can learn to add realistic colors to black and white images.

## 📋 Table of Contents
- [About](#-about)
- [Features](#-features)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Training](#-training)
- [Results](#-results)
- [How It Works](#-how-it-works)

## 🎯 About

This project implements a **Conditional GAN** for **grayscale-to-color image translation**. The model learns to colorize grayscale flower images by training on paired data where:
- **Input**: Grayscale version of flower images
- **Target**: Original colored flower images

The implementation uses a **U-Net Generator** with skip connections and a **PatchGAN Discriminator** for high-quality image translation.

## ✨ Features

- 🌸 **Flower Image Colorization** - Converts grayscale flowers to realistic colored images
- 🏗️ **U-Net Generator** - Deep encoder-decoder architecture with skip connections
- 🛡️ **PatchGAN Discriminator** - Evaluates image patches for realistic details
- 📊 **TensorFlow Flowers Dataset** - Uses built-in TensorFlow dataset (200 samples)
- 🎨 **Real-time Visualization** - Shows training progress with generated images
- ⚡ **Fast Training** - Optimized for quick experimentation (3 epochs)
- 🔧 **Customizable Parameters** - Easy to modify batch size, epochs, and loss weights

## 🛠️ Installation

**Option 1: Google Colab (Recommended)**
1. Click the "Open in Colab" button above
2. Run all cells in order - dependencies install automatically!

**Option 2: Local Setup**
```bash
# Clone the repository
git clone https://github.com/KritikMahesh/PRODIGY_GA_04.git
cd PRODIGY_GA_04

# Install dependencies
pip install tensorflow matplotlib tensorflow-datasets numpy

# Run the notebook
jupyter notebook Image_to_Image_Translation_with_cGAN.ipynb
```

## 📊 Dataset

**TensorFlow Flowers Dataset**
- 🌻 **Source**: Built-in TensorFlow Datasets
- 📸 **Images**: 3,670 flower photos (5 classes)
- 🎯 **Usage**: 200 samples for quick training
- 📏 **Size**: Resized to 256x256 pixels
- 🎨 **Processing**: Automatic grayscale conversion for input

```python
# Dataset preprocessing pipeline
def preprocess(image):
    image = tf.image.resize(image, [256, 256])
    input_image = rgb_to_grayscale(image)  # Convert to grayscale
    target_image = image                   # Keep original colors
    return normalize(input_image, target_image)
```

## 🚀 Usage

### Quick Start
1. **Load the notebook** and run all cells sequentially
2. **Watch the training** - loss values print every 50 steps
3. **See results** - generated colorized images appear after training
4. **Experiment** - modify EPOCHS, batch size, or dataset size

### Key Parameters
```python
EPOCHS = 3              # Number of training epochs
BATCH_SIZE = 4          # Batch size for training
IMAGE_SIZE = 256        # Input/output image dimensions
LAMBDA = 100            # L1 loss weight (reconstruction)
LEARNING_RATE = 2e-4    # Adam optimizer learning rate
BETA_1 = 0.5           # Adam optimizer beta1 parameter
```

## 🏗️ Model Architecture

### 🎨 Generator (U-Net)
- **Architecture**: Encoder-Decoder with skip connections
- **Input**: 256x256x3 grayscale image (3 channels for compatibility)
- **Output**: 256x256x3 colorized image
- **Layers**: 8 downsampling + 7 upsampling layers
- **Skip Connections**: Preserve fine details during upsampling

```python
# Generator structure
down_stack = [
    downsample(64, 4, apply_batchnorm=False),   # 128x128x64
    downsample(128, 4),                         # 64x64x128
    downsample(256, 4),                         # 32x32x256
    downsample(512, 4),                         # 16x16x512
    # ... more layers
]

up_stack = [
    upsample(512, 4, apply_dropout=True),       # 2x2x1024
    upsample(512, 4, apply_dropout=True),       # 4x4x1024
    # ... more layers with skip connections
]
```

### 🛡️ Discriminator (PatchGAN)
- **Architecture**: Convolutional classifier
- **Input**: Concatenated input + target images (256x256x6)
- **Output**: 30x30x1 patch predictions
- **Purpose**: Evaluates if each image patch looks realistic

```python
# Discriminator evaluates image pairs
x = layers.concatenate([input_image, target_image])
# ... convolutional layers
# Output: 30x30x1 (real/fake for each patch)
```

## 🎯 Training

### Loss Functions
1. **Generator Loss** = Adversarial Loss + L1 Reconstruction Loss
   ```python
   gan_loss = loss_object(tf.ones_like(disc_output), disc_output)
   l1_loss = tf.reduce_mean(tf.abs(target - generated))
   total_loss = gan_loss + (100 * l1_loss)  # L1 weighted by 100
   ```

2. **Discriminator Loss** = Real Loss + Fake Loss
   ```python
   real_loss = loss_object(tf.ones_like(real_output), real_output)
   fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
   total_loss = real_loss + fake_loss
   ```

### Training Process
- **Optimizer**: Adam (lr=2e-4, beta1=0.5)
- **Steps per Epoch**: ~50 (200 samples ÷ 4 batch size)
- **Progress Tracking**: Loss printed every 50 steps
- **Duration**: ~2-3 minutes per epoch on GPU

## 📈 Results

The model generates colorized images showing:
- **Input Image**: Grayscale flower
- **Ground Truth**: Original colored flower  
- **Predicted Image**: AI-generated colorized version

### Expected Performance
- **Training Time**: ~5-10 minutes total (3 epochs)
- **Generator Loss**: Decreases from ~100 to ~20-30
- **Discriminator Loss**: Stabilizes around 0.5-1.5
- **Visual Quality**: Realistic colors with good detail preservation

## 🔬 How It Works

### 🎨 **Generator Process**
1. **Input**: Takes grayscale flower image (256x256x3)
2. **Encoding**: Downsamples through 8 convolutional layers
3. **Bottleneck**: Compressed representation (1x1x512)
4. **Decoding**: Upsamples through 7 layers with skip connections
5. **Output**: Colorized image (256x256x3) with tanh activation

### 🛡️ **Discriminator Process**
1. **Input**: Concatenates grayscale + color image pair
2. **Evaluation**: Processes through convolutional layers
3. **Output**: 30x30 grid of real/fake predictions (PatchGAN)
4. **Training**: Learns to distinguish real vs generated pairs

### ⚔️ **Adversarial Training**
- **Generator Goal**: Fool discriminator + match target colors
- **Discriminator Goal**: Detect fake colorized images
- **Balance**: Both networks improve together over iterations
- **Result**: Generator learns realistic colorization patterns

## 🔧 Customization

### Modify Training Parameters
```python
EPOCHS = 10           # Train longer for better results
BATCH_SIZE = 8        # Increase if you have more GPU memory
LAMBDA = 50           # Reduce L1 weight for more creative colors
```

### Use Different Dataset
```python
# Replace tf_flowers with other datasets
dataset, info = tfds.load('your_dataset', with_info=True, as_supervised=True)
```

---
