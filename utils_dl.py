import os
import cv2
import torch
import numpy as np
from config import config
from pathlib import Path
from g1_model import EdgeGenerator


def apply_canny(image):
    """
    Apply Canny edge detection to an image.
    """
    # Ensure image is in the right format for Canny
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
            
    # Apply Canny edge detection
    edges = cv2.Canny(image, config.CANNY_THRESHOLD_LOW, config.CANNY_THRESHOLD_HIGH) # Shape: (H, W)
    
    # Invert and normalize to [0, 1] for the edge map
    edges = (255 - edges).astype(np.float32) / 255.0
    return edges

def dilate_mask(mask, kernel_size=5, iterations=1):
    """
    Dilate the binary mask.
    
    Args:
        mask: Input mask tensor or numpy array. Should be in format where:
            0 = missing pixels (value < 10)
            255 = known pixels (value >= 10)
        kernel_size: Size of dilation kernel
        iterations: Number of dilation iterations
    
    Returns:
        Dilated mask as a numpy array where:
        1.0 = missing pixels
        0.0 = known pixels
    """
    # Convert mask to numpy if it's a tensor
    if isinstance(mask, torch.Tensor):
        mask_np = mask.squeeze().cpu().numpy()  # Shape: (H, W)
    else:
        mask_np = mask.squeeze() if hasattr(mask, 'squeeze') else mask
    
    # Make sure mask is in the proper format [0-255] with 0 for missing pixels
    # If mask is in [0,1] range, convert to [0,255]
    if mask_np.max() <= 1.0 and np.min(mask_np) >= 0:
        mask_np = mask_np * 255

    # Binary mask: 1 for missing pixels (where mask == 0), 0 for known pixels
    binary_mask = (mask_np < 10).astype(np.float32)
    
    # Dilate the binary mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
    dilated_mask_uint8 = cv2.dilate(binary_mask_uint8, kernel, iterations=iterations)
    
    # Convert to [0.0, 1.0] range where 1.0 = missing pixels
    dilated_mask = dilated_mask_uint8.astype(np.float32) / 255.0
    
    return dilated_mask

def remove_mask_edge(mask, img):
    """
    Remove the edges of the mask in the edge image by painting white (1.0)
    where the dilated mask indicates missing regions.
    
    Args:
        mask: Input mask tensor or numpy array
        img: Edge image tensor or numpy array
    
    Returns:
        Edge image with mask edges removed
    """
    # Convert img to numpy if it's a tensor
    if isinstance(img, torch.Tensor):
        img_np = img.squeeze().cpu().numpy()  # Shape: (H, W)
    else:
        img_np = img.squeeze() if hasattr(img, 'squeeze') else img
    
    # Get dilated mask
    dilated_mask = dilate_mask(mask)
    
    # Paint white (1.0) where dilated mask indicates missing pixels
    result = np.where(dilated_mask > 0.5, 1.0, img_np)
    
    return result

def gen_raw_mask(input_img):
    # Extract mask: Consider pixels as missing if all RGB values > 245
    mask_binary = np.all(input_img > 245, axis=-1).astype(np.float32)  # Shape: (H, W)
    raw_mask = 255 - mask_binary * 255  # Invert mask (0s for missing pixels, 255s for known pixels)
    return raw_mask  # Shape: (H, W)


###################################################
# G2 dataloader functions
####################################################


def gen_gidance_img(input_img, edge_img, edge_color=(0, 0, 0)):
    """
    Generate a guidance image by:
    1. Using TELEA inpainting on masked regions.
    2. Overlaying colored edges (e.g., red) across the entire image.

    Args:
        input_img (np.ndarray): Masked BGR image (H, W, 3)
        edge_img (np.ndarray): Grayscale predicted edge image (H, W)
        edge_threshold (int): Edge threshold for overlay
        edge_color (tuple): BGR tuple for edge overlay color

    Returns:
        np.ndarray: Guidance image ready for G2 input
    """
    # Step 1: Generate binary mask from white pixels using provided utility
    # Step 2: Inpaint the image using TELEA after dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expanded_mask = cv2.dilate((gen_raw_mask(input_img) < 10).astype(np.uint8), kernel, iterations=1)

    inpaint_input = input_img.copy()
    inpaint_input[expanded_mask == 1] = 0
    inpainted_color = cv2.inpaint(inpaint_input, expanded_mask * 255, 15, cv2.INPAINT_TELEA)

    # Step 3: Overlay edge map across the entire image - not just masked regions
    all_edges = (edge_img < 150)

    guidance_img = inpainted_color.copy()
    guidance_img[all_edges] = edge_color

    return guidance_img


def validate_edge_map(split="train"):
    """
    Validates if the number of images in the input folder matches the number of images in the edge folder.
    If the number of files doesn't match or any edge map is missing, the edge folder is cleared, and edge maps are regenerated using the _generate_edge_maps function.

    Args:
        split (str): Dataset split to use ('train', 'test', or 'val').

    Returns:
        bool: True if validation and regeneration (if needed) are successful, False otherwise.
    """
    # Select input and edge directories based on the split
    if split == "train":
        input_dir = config.TRAIN_IMAGES_INPUT
        edge_dir = config.TRAIN_EDGE_DIR
    elif split == "test":
        input_dir = config.TEST_IMAGES_INPUT
        edge_dir = config.TEST_EDGE_DIR
    elif split == "val":
        input_dir = config.VAL_IMAGES_INPUT
        edge_dir = config.VAL_EDGE_DIR
    else:
        raise ValueError("Invalid split. Choose from 'train', 'test', or 'val'.")

    # Ensure edge directory exists
    os.makedirs(edge_dir, exist_ok=True)

    input_path = Path(input_dir)
    input_files = sorted([f for f in input_path.glob("*.jpg")])
    edge_files = sorted([f.name for f in os.scandir(edge_dir) if f.name.endswith('.jpg')])

    # Check if the number of files matches
    if len(input_files) != len(edge_files):
        print(f"Mismatch in number of images: {len(input_files)} input images vs {len(edge_files)} edge images.")
        print("Clearing edge folder and regenerating edge maps...")
        _clear_folder(edge_dir)
        _generate_edge_maps(split=split, batch_size=config.BATCH_SIZE_G1_INFERENCE)
        return False

    # Check if all input images have corresponding edge maps
    for input_file in input_files:
        # Change this line to match the actual file naming in _generate_edge_maps
        expected_edge_file = f"{os.path.splitext(input_file.name)[0]}_edge_map.jpg"
        if expected_edge_file not in edge_files:
            print(f"Missing edge map for {input_file.name}. Clearing edge folder and regenerating edge maps...")
            _clear_folder(edge_dir)
            _generate_edge_maps(split=split, batch_size=config.BATCH_SIZE_G1_INFERENCE)
            return False

    print("Number of images and corresponding edge maps match.")
    return True


def validate_guidance_images(split="train"):
    """
    Validates if guidance images exist for all input images.
    If any guidance image is missing, they will be generated.

    Args:
        split (str): Dataset split to use ('train', 'test', or 'val').

    Returns:
        bool: True if validation was successful, False if regeneration was needed.
    """
    # Select directories based on the split
    if split == "train":
        input_dir = config.TRAIN_IMAGES_INPUT
        guidance_dir = config.TRAIN_GUIDANCE_DIR
    elif split == "test":
        input_dir = config.TEST_IMAGES_INPUT
        guidance_dir = config.TEST_GUIDANCE_DIR
    elif split == "val":
        input_dir = config.VAL_IMAGES_INPUT
        guidance_dir = config.VAL_GUIDANCE_DIR
    else:
        raise ValueError("Invalid split. Choose from 'train', 'test', or 'val'.")

    # Ensure guidance directory exists
    os.makedirs(guidance_dir, exist_ok=True)
    
    # Get list of input and guidance images
    input_path = Path(input_dir)
    guidance_path = Path(guidance_dir)
    input_files = sorted([f.name for f in input_path.glob("*.jpg")])
    guidance_files = sorted([f.name for f in guidance_path.glob("*.jpg")])
    
    # Check if numbers match
    if len(input_files) != len(guidance_files):
        print(f"Mismatch in number of images: {len(input_files)} input images vs {len(guidance_files)} guidance images.")
        print("Generating missing guidance images...")
        _generate_guidance_images(split=split)
        return False
    
    # Check if each input has a corresponding guidance image
    for input_file in input_files:
        basename = os.path.splitext(input_file)[0]
        expected_guidance_file = f"{basename}.jpg"
        if expected_guidance_file not in guidance_files:
            print(f"Missing guidance image for {input_file}. Generating guidance images...")
            _generate_guidance_images(split=split)
            return False
    
    print("All guidance images exist.")
    return True


def _clear_folder(folder_path):
    """
    Clears all files in the specified folder.

    Args:
        folder_path (str): Path to the folder to clear.
    """
    for file in os.scandir(folder_path):
        os.remove(file.path)
    print(f"Cleared folder: {folder_path}")


def _generate_edge_maps(split="train", batch_size=config.BATCH_SIZE_G1_INFERENCE):
    """
    Generates edge maps for all input images in batches and saves them in the edge folder.
    """
    # Import here instead of at the top level
    from dataloader_g1 import get_dataloader_g1
    
    # Select input and edge directories based on the split
    if split == "train":
        edge_dir = config.TRAIN_EDGE_DIR
    elif split == "test":
        edge_dir = config.TEST_EDGE_DIR
    elif split == "val":
        edge_dir = config.VAL_EDGE_DIR
    else:
        raise ValueError("Invalid split. Choose from 'train', 'test', or 'val'.")

    # Ensure the edge directory exists
    os.makedirs(edge_dir, exist_ok=True)

    # Load the checkpoint
    checkpoint_path = config.G1_MODEL_PATH  # Path to the G1 model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

    # Initialize the model architecture
    model = EdgeGenerator()

    # Check if the checkpoint contains a full dictionary or just the state_dict
    if "g1_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["g1_state_dict"])  # Load only the model weights
    else:
        model.load_state_dict(checkpoint)  # Directly load raw weights if no wrapper exists

    model.eval()  # Set the model to evaluation mode
    model.to(config.DEVICE)  # Move the model to the specified device (e.g., GPU)

    # Initialize the dataloader with use_gt=False and filenames=True
    dataloader = get_dataloader_g1(split=split, batch_size=config.BATCH_SIZE_G1_INFERENCE ,use_mask=True, use_gt=False, return_filenames=True)

    # Process images in batches
    for batch_idx, batch in enumerate(dataloader):
        input_edge = batch["input_edge"].to(config.DEVICE)  # Shape: (batch_size, 1, H, W)
        gray = batch["gray"].to(config.DEVICE)              # Shape: (batch_size, 1, H, W)
        mask = batch["mask"].to(config.DEVICE)              # Shape: (1, H, W)
        filenames = batch.get("filenames", None)            # Get filenames if available

        # Generate edge maps for the batch
        with torch.no_grad():
            edge_maps = model(input_edge, mask, gray)  # Pass the batch through the model

        # Save the generated edge maps
        for j in range(edge_maps.size(0)):
            edge_map = edge_maps[j].cpu().numpy().squeeze()  # Convert to numpy and remove channel dimension
            edge_map = (edge_map * 255).astype(np.uint8)     # Scale to [0, 255]

            # Use original filename if available, otherwise fallback to index
            if filenames is not None:
                # Extract number from filename (assuming format like "1234.jpg")
                basename = os.path.splitext(filenames[j])[0]
                edge_path = os.path.join(edge_dir, f"{basename}_edge_map.jpg")
            else:
                edge_path = os.path.join(edge_dir, f"edge_map_{batch_idx * batch_size + j}.jpg")

            cv2.imwrite(edge_path, edge_map)
        
        if batch_idx % 10 == 0:
            print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

    print(f"Generated edge maps for all input images in {edge_dir}.")
    
    # Clean up GPU memory
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _process_single_guidance_image(args):
    """
    Process a single image for guidance generation.
    
    Args:
        args: Tuple containing (file, input_dir, edge_dir, guidance_dir)
        
    Returns:
        bool: True if successful, False otherwise
    """
    file, input_dir, edge_dir, guidance_dir = args
    basename = os.path.splitext(file.name)[0]
    input_path = str(file)  # Convert WindowsPath to string
    edge_path = os.path.join(edge_dir, f"{basename}_edge_map.jpg")
    guidance_path = os.path.join(guidance_dir, file.name)
    
    # Skip if guidance image already exists
    if os.path.exists(guidance_path):
        return False
        
    # Read images
    input_img = cv2.imread(input_path)
    edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    
    if input_img is None or edge_img is None:
        print(f"Warning: Could not read input or edge image for {file.name}")
        return False
    
    # Generate guidance image using input and edge
    guidance_img = gen_gidance_img(input_img, edge_img)
    
    # Save the guidance image
    cv2.imwrite(guidance_path, guidance_img)
    return True


def _generate_guidance_images(split="train", num_workers=16):
    """
    Generates guidance images for all input images based on edge maps using parallel processing.
    
    Args:
        split (str): Dataset split to use ('train', 'test', or 'val').
        num_workers (int): Number of worker processes to use.
    """
    import concurrent.futures
    from tqdm import tqdm
    
    # Select directories based on the split
    if split == "train":
        input_dir = config.TRAIN_IMAGES_INPUT
        guidance_dir = config.TRAIN_GUIDANCE_DIR
        edge_dir = config.TRAIN_EDGE_DIR
    elif split == "test":
        input_dir = config.TEST_IMAGES_INPUT
        guidance_dir = config.TEST_GUIDANCE_DIR
        edge_dir = config.TEST_EDGE_DIR
    elif split == "val":
        input_dir = config.VAL_IMAGES_INPUT
        guidance_dir = config.VAL_GUIDANCE_DIR
        edge_dir = config.VAL_EDGE_DIR
    else:
        raise ValueError("Invalid split. Choose from 'train', 'test', or 'val'.")
    
    # First make sure edge maps exist
    validate_edge_map(split)
    
    # Ensure guidance directory exists
    os.makedirs(guidance_dir, exist_ok=True)
    
    # Get all input images
    input_path = Path(input_dir)
    input_files = sorted([f for f in input_path.glob("*.jpg")])
    total_files = len(input_files)
    
    print(f"Generating guidance images for {total_files} input images using {num_workers} workers...")
    
    # Prepare argument tuples for each task
    tasks = [(file, input_dir, edge_dir, guidance_dir) for file in input_files]
    
    # Process images in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to display progress
        completed = 0
        failed = 0
        
        # Process all tasks with progress tracking
        results = list(tqdm(
            executor.map(_process_single_guidance_image, tasks), 
            total=total_files, 
            desc="Generating guidance images"
        ))
        
        # Count successes and failures
        for success in results:
            if success:
                completed += 1
            else:
                failed += 1
    
    print(f"Generated {completed} guidance images in {guidance_dir}. Failed: {failed}.")

