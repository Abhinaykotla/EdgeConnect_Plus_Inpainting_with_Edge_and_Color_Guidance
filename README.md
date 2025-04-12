
# EdgeConnect+ : Adversarial Inpainting with Edge and Color Guidance

This repository contains the implementation of **EdgeConnect+**, a deep learning-based image inpainting model that extends the original [EdgeConnect](https://arxiv.org/abs/1901.00212) architecture. EdgeConnect+ integrates both **structural guidance (edges)** and **chromatic guidance (blurred color map)** to reconstruct missing regions of images in a visually realistic and perceptually consistent manner.

## 📌 Project Overview

EdgeConnect+ follows a **three-stage inpainting pipeline**:

1. **Edge Generation (G1)**: Predicts missing edge structures from a grayscale masked input.
2. **Color Guidance**: Provides low-frequency color hints by applying a Gaussian blur to the original image. We plan to extend this by exploring Partial Convolutions and Contextual Attention modules.
3. **Final Inpainting (G2)** *(In Progress)*: Generates the completed image using a composite input of predicted edges, blurred color map, and the original image content outside the mask.

---

## ✅ Current Progress

### ✔ G1: Edge Generator

- Trained on the CelebA dataset (256x256 resolution).
- Utilizes L1, adversarial, and feature-matching losses.
- Stable training with strong edge predictions (see `generated_samples/` for visual outputs).
- Edge predictions generated from masked grayscale input + Canny preprocessing.

### ✔ Dataset Preparation

- Using **CelebA dataset** with center-cropped and resized images.
- Irregular masks covering at least 20% of each image.
- Grayscale and Canny edge preprocessing applied.
- Ground truth edges are generated for supervision.

### ✔ Evaluation Metrics Planned

- **PSNR**, **SSIM**, **L1**, and **LPIPS** for evaluating reconstruction performance.

### 🚧 G2: Final Inpainting Network

- In progress.
- Will use U-Net or encoder-decoder architecture.
- Takes composite RGB input (edges + color map) + binary mask.
- Training plan includes perceptual, style, and adversarial losses.

---

## 📁 Project Structure

```
├── configs/               # Config files and hyperparameters
├── data/                  # Preprocessed CelebA images and masks
├── models/                # G1, G2, and discriminator implementations
├── train.py               # Main training loop
├── utils.py               # Utility functions for preprocessing and evaluation
├── notebooks/             # Jupyter notebooks for data visualization
├── results/               # Output samples and loss curves
└── README.md              # Project overview and updates
```

---

## 🔧 Next Steps

- Finalize training of G1 (continue for more epochs).
- Begin training G2 using the composite edge + color + mask input.
- Integrate advanced color filling mechanisms.
- Run full pipeline and evaluate on held-out test images.

---

## 📊 Visual Results

Example visualizations are available in `results/`, including edge maps, masked inputs, and G1 predictions.

---

## 🤝 Acknowledgments

- [EdgeConnect (Nazeri et al.)](https://arxiv.org/abs/1901.00212)
- [PartialConv (Liu et al.)](https://arxiv.org/abs/1804.07723)
- [Contextual Attention (Yu et al.)](https://arxiv.org/abs/1801.07892)
- CelebA Dataset from MMLAB

---

## 🧪 Citation

If using this codebase, please cite the original EdgeConnect paper and acknowledge EdgeConnect+ as an experimental extension.

---

Stay tuned for updates on the G2 implementation and complete pipeline results!
