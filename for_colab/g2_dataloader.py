import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_mask(image):
    return np.all(image > 245, axis=-1).astype(np.uint8)  # white = 1, else 0

def apply_guided_mask_blur(masked_image, mask):
    """
    Expands and removes white masked regions, then uses TELEA to fill naturally.
    """
    image = masked_image.copy().astype(np.uint8)

    # Dilate mask slightly to avoid white border artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    # Zero out masked region before inpainting
    image[expanded_mask == 1] = [0, 0, 0]

    # Inpaint using TELEA
    mask_8u = (expanded_mask * 255).astype(np.uint8)
    inpainted = cv2.inpaint(image, mask_8u, 15, cv2.INPAINT_TELEA)

    return inpainted

def generate_color_map(masked_image, mask):
    return apply_guided_mask_blur(masked_image, mask)

def overlay_edges_only_in_mask(image, edge_map, mask, threshold=30, edge_color=(255, 0, 0)):
    """
    Draws thin red edges (default: red in BGR) only within the masked regions.
    """
    edge_binary = (edge_map > threshold).astype(np.uint8)
    result = image.copy()

    # Vectorized logic: where edge exists inside the mask
    masked_edges = np.logical_and(mask == 1, edge_binary == 0)
    result[masked_edges] = edge_color  # Apply red only inside masked edge pixels

    return result

def visualize_color_processing(masked_img_path, edge_img_path, image_size=256):
    input_img = cv2.imread(masked_img_path)
    edge_img = cv2.imread(edge_img_path, cv2.IMREAD_GRAYSCALE)

    input_img = cv2.resize(input_img, (image_size, image_size))
    edge_img = cv2.resize(edge_img, (image_size, image_size))

    mask = extract_mask(input_img)

    # Step 1: Generate inpainted color map
    color_map = generate_color_map(input_img, mask)

    # Step 2: Overlay thin red edges
    final_result = overlay_edges_only_in_mask(color_map, edge_img, mask, edge_color=(255, 0, 0))

    # Plot results
    fig, axs = plt.subplots(1, 5, figsize=(24, 6))
    axs[0].imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Input Masked Image")
    axs[1].imshow(edge_img, cmap="gray")
    axs[1].set_title("Predicted Edge")
    axs[2].imshow(cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Inpainted Color Map")
    axs[3].imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
    axs[3].set_title("Blue Edges Overlaid")
    axs[4].imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
    axs[4].set_title("Final Output")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    masked_path = "/content/drive/MyDrive/edgeconnect/data_archive/CelebA/test_input/000024.jpg"
    edge_path = "/content/drive/MyDrive/edgeconnect/results/output_edge_000024.png"
    visualize_color_processing(masked_path, edge_path)
