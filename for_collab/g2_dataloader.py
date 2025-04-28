
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

def apply_enhanced_blur(image, kernel_size=51, sigma=20, enhance=True, boost_saturation=True, gamma=1.2):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    if enhance:
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        blurred = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if boost_saturation:
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * 2.0, 0, 255)
        hsv = cv2.merge((h, s, v)).astype(np.uint8)
        blurred = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
        blurred = cv2.LUT(blurred, table)

    return blurred

def extract_mask(image):
    mask = np.all(image > 245, axis=-1).astype(np.uint8)
    return mask

def combine_blur_with_original(original, blurred, mask, feather_kernel=31, feather_sigma=10):
    mask = mask.astype(np.float32)
    feathered_mask = cv2.GaussianBlur(mask, (feather_kernel, feather_kernel), feather_sigma)
    feathered_mask = feathered_mask[:, :, None]
    return original * (1 - feathered_mask) + blurred * feathered_mask

def process_image(masked_img_path, edge_img_path, image_size=256):
    input_img = cv2.imread(masked_img_path)  # BGR
    edge_img = cv2.imread(edge_img_path, cv2.IMREAD_GRAYSCALE)

    input_img = cv2.resize(input_img, (image_size, image_size))
    edge_img = cv2.resize(edge_img, (image_size, image_size))

    input_img = input_img.astype(np.float32) / 255.0
    edge_img = edge_img.astype(np.float32) / 255.0

    mask = extract_mask((input_img * 255).astype(np.uint8))
    blurred = apply_enhanced_blur((input_img * 255).astype(np.uint8))
    blurred = blurred.astype(np.float32) / 255.0
    blended_img = combine_blur_with_original(input_img, blurred, mask)

    input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1))
    edge_tensor = torch.from_numpy(edge_img).unsqueeze(0)
    blur_tensor = torch.from_numpy(blended_img.transpose(2, 0, 1))
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

    return {
        "input_img": input_tensor,
        "edge": edge_tensor,
        "blur": blur_tensor,
        "mask": mask_tensor
    }

def visualize_image(masked_img_path, edge_img_path):
    data = process_image(masked_img_path, edge_img_path)

    input_img = data["input_img"].permute(1, 2, 0).numpy()
    edge_img = data["edge"].squeeze().numpy()
    blur_img = data["blur"].permute(1, 2, 0).numpy()
    mask_img = data["mask"].squeeze().numpy()

    input_img = input_img[:, :, ::-1]
    blur_img = blur_img[:, :, ::-1]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(input_img)
    axs[0].set_title("Input Masked Image")
    axs[1].imshow(edge_img, cmap="gray")
    axs[1].set_title("Edge Map from G1")
    axs[2].imshow(mask_img, cmap="gray")
    axs[2].set_title("Extracted Mask")
    axs[3].imshow(blur_img)
    axs[3].set_title("Strongly Enhanced Color-Guided")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    masked_img_path = "/content/drive/MyDrive/edgeconnect/data_archive/CelebA/test_input/000024.jpg"
    edge_img_path = "/content/drive/MyDrive/edgeconnect/00024.png"
    visualize_image(masked_img_path, edge_img_path)
