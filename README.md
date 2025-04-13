# EdgeConnect+: Adversarial Inpainting with Edge and Color Guidance

## Overview
**EdgeConnect+** is an advanced image inpainting framework that enhances the original EdgeConnect model by integrating both **edge** and **color guidance** for realistic image restoration. Our approach aims to improve the **structural accuracy** and **color consistency** of inpainted images, making it suitable for applications in **photo restoration, object removal, and content-aware image editing**.

## Features
- **Edge-Driven Inpainting**: Uses an edge generator to predict missing structures.
- **Color Guidance via Gaussian Blur**: Incorporates a smoothed color map to ensure seamless blending.
- **GAN-Based Architecture**: Leverages generative adversarial networks (GANs) to enhance realism.
- **Multi-Dataset Training**: Evaluated on CelebA (faces) and Places2 (scenes) for diverse inpainting capabilities.

## Methodology
1. **Edge Generation**: A GAN-based edge generator predicts missing edges.
2. **Color Map Generation**: A Gaussian-blurred color map provides contextual hints for missing regions.
3. **Final Image Reconstruction**: A second GAN synthesizes the final inpainted image using edge and color information.

## Datasets
- **[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)**: 202,599 facial images with various poses and expressions.
- **[Places2](http://places2.csail.mit.edu/)**: 1.8 million scene images across 365 categories.

<!-- ## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/EdgeConnect-Plus.git
   cd EdgeConnect-Plus
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download datasets (optional):
   ```sh
   bash scripts/download_datasets.sh
   ```

## Usage
### Training the Model
```sh
python train.py --dataset celebA --epochs 50
```
### Testing the Model
```sh
python test.py --input sample.jpg --output result.jpg
```
### Visualization
```sh
python visualize.py --dataset places2
``` -->

## Contributors
- **Abhinay Kotla** (axk5827@mavs.uta.edu)
- **Sanjana Ravi Prakash** (sxr8375@mavs.uta.edu)

## License
This project is licensed under the MIT License.

---
