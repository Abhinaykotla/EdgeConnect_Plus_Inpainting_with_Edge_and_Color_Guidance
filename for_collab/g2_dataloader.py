import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_mask(image):
    return np.all(image > 245, axis=-1).astype(np.uint8)  # white = 1, else 0

def apply_guided_mask_blur(masked_image, mask):
    """
    Removes masked (white) regions and uses TELEA to fill them in.
    """
    image = masked_image.copy().astype(np.uint8)

    # Dilate mask to eliminate white edge borders
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    # Remove white from image
    image[expanded_mask == 1] = [0, 0, 0]

    # Inpaint with TELEA
    mask_8u = (expanded_mask * 255).astype(np.uint8)
    inpainted = cv2.inpaint(image, mask_8u, 15, cv2.INPAINT_TELEA)

    return inpainted

def generate_color_map(masked_image, mask):
    return apply_guided_mask_blur(masked_image, mask)

def overlay_edges_only_in_mask(image, edge_map, mask, threshold=30, edge_color=(0, 0, 255)):
    """
    Draws thin colored edges (default: blue) only inside masked regions.
    """
    edge_binary = (edge_map > threshold).astype(np.uint8)
    result = image.copy()

    for y in range(edge_binary.shape[0]):
        for x in range(edge_binary.shape[1]):
            if mask[y, x] == 1 and edge_binary[y, x] == 0:
                result[y, x] = edge_color
    return result

def visualize_color_processing(masked_img_path, edge_img_path, image_size=256):
    input_img = cv2.imread(masked_img_path)
    edge_img = cv2.imread(edge_img_path, cv2.IMREAD_GRAYSCALE)

    input_img = cv2.resize(input_img, (image_size, image_size))
    edge_img = cv2.resize(edge_img, (image_size, image_size))

    mask = extract_mask(input_img)

    # Step 1: Generate color map
    color_map = generate_color_map(input_img, mask)

    # Step 2: Inpaint the image
    inpainted = color_map.copy()  # Already inpainted from previous step

    # Step 3: Apply blue edges AFTER inpainting
    final_result = overlay_edges_only_in_mask(inpainted, edge_img, mask, edge_color=(0, 0, 255))

    # Plot
    fig, axs = plt.subplots(1, 5, figsize=(24, 6))
    axs[0].imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Input Masked Image")
    axs[1].imshow(edge_img, cmap="gray")
    axs[1].set_title("Predicted Edge")
    axs[2].imshow(cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Color Map (Inpainted)")
    axs[3].imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
    axs[3].set_title("Final Enhanced with Blue Edges")
    axs[4].imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
    axs[4].set_title("Post-Inpainted (TELEA + Blue Edges)")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    masked_path = "/content/drive/MyDrive/edgeconnect/data_archive/CelebA/test_input/000007.jpg"
    edge_path = "/content/drive/MyDrive/edgeconnect/results/edge_output.png"
    visualize_color_processing(masked_path, edge_path)
