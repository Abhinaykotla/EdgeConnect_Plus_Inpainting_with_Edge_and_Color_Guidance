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
    
    Converts the image to grayscale if needed and applies the Canny edge detection algorithm
    with thresholds specified in the config. Returns an inverted, normalized edge map.
    
    Args:
        image (numpy.ndarray): Input image, can be grayscale or color
        
    Returns:
        numpy.ndarray: Edge map with values in range [0, 1] where 1 represents non-edges Shape: (H, W)
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
    
    Expands the masked (hole) regions in the image by applying morphological dilation.
    This is useful to ensure edges around holes are properly masked out.
    
    Args:
        mask (torch.Tensor or numpy.ndarray): Input mask where:
            0 = missing pixels (value < 10)
            255 = known pixels (value >= 10)
        kernel_size (int): Size of dilation kernel, controls dilation amount
        iterations (int): Number of dilation iterations, controls dilation intensity
    
    Returns:
        numpy.ndarray: Dilated mask as a numpy array where:
        1.0 = missing pixels (holes)
        0.0 = known pixels (valid regions)
        Shape: (H, W)
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
    
    This ensures that edges don't appear at the boundaries of the masked regions,
    which can cause artifacts in the inpainting process.
    
    Args:
        mask (torch.Tensor or numpy.ndarray): Input mask
        img (torch.Tensor or numpy.ndarray): Edge image to clean up
    
    Returns:
        numpy.ndarray: Edge image with mask edges removed (white/1.0 in masked regions)
        Shape: (H, W)
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
    """
    Generate a binary mask from an image by thresholding white pixels.
    
    Assumes that white pixels (RGB > 245) represent holes in the image.
    Returns an inverted mask where 0 represents holes and 255 represents valid regions.
    
    Args:
        input_img (numpy.ndarray): RGB input image, shape (H, W, 3)
    
    Returns:
        numpy.ndarray: Binary mask where:
            0 = missing pixels (holes)
            255 = known pixels (valid regions)
        Shape: (H, W)
    """
    # Extract mask: Consider pixels as missing if all RGB values > 245
    mask_binary = np.all(input_img > 245, axis=-1).astype(np.float32)  # Shape: (H, W)
    raw_mask = 255 - mask_binary * 255  # Invert mask (0s for missing pixels, 255s for known pixels)
    return raw_mask  # Shape: (H, W)


###################################################
# G2 dataloader functions
####################################################


def gen_guidance_img(input_img, edge_img, edge_color=(0, 0, 0)):
    """
    Generate a guidance image by:
    1. Using TELEA inpainting on masked regions.
    2. Overlaying colored edges (e.g., black) across the entire image.
    
    This creates a guidance image that combines rough color information (from inpainting)
    with structural information (from edges) to guide the G2 inpainting network.
    
    Args:
        input_img (np.ndarray): Masked BGR image, shape (H, W, 3)
        edge_img (np.ndarray): Grayscale predicted edge image, shape (H, W)
        edge_color (tuple): BGR tuple for edge overlay color, default is black (0,0,0)

    Returns:
        np.ndarray: Guidance image ready for G2 input, shape (H, W, 3)
    """
    # Step 1: Generate binary mask from white pixels using provided utility
    # Step 2: Inpaint the image using TELEA after dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expanded_mask = cv2.dilate((gen_raw_mask(input_img) < 10).astype(np.uint8), kernel, iterations=1)

    # Create inpainting input and fill holes with fast marching method (TELEA)
    inpaint_input = input_img.copy()
    inpaint_input[expanded_mask == 1] = 0  # Set masked regions to black for inpainting
    inpainted_color = cv2.inpaint(inpaint_input, expanded_mask * 255, 15, cv2.INPAINT_TELEA)

    # Step 3: Overlay edge map across the entire image - not just masked regions
    # Threshold edge map to get binary edges (edge_img < 150 represents edge pixels)
    all_edges = (edge_img < 150)

    guidance_img = inpainted_color.copy()
    guidance_img[all_edges] = edge_color  # Apply edge overlay with specified color

    return guidance_img


def validate_edge_map(split="train"):
    """
    Validates if the number of images in the input folder matches the number of images in the edge folder.
    If the number of files doesn't match or any edge map is missing, the edge folder is cleared, and edge maps are regenerated.
    
    This ensures all input images have corresponding edge maps before training or inference.
    
    Args:
        split (str): Dataset split to use ('train', 'test', 'val', or 'demo').

    Returns:
        bool: True if validation successful (all edge maps exist), False if regeneration was needed.
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
    elif split == "demo":
        input_dir = config.DEMO_IMAGES_INPUT
        edge_dir = config.DEMO_EDGE_DIR
    else:
        raise ValueError("Invalid split. Choose from 'train', 'test', 'val', or 'demo'.")

    # Ensure edge directory exists
    os.makedirs(edge_dir, exist_ok=True)

    input_path = Path(input_dir)
    input_files = sorted([f for f in input_path.glob("*.jpg")])
    edge_files = sorted([f.name for f in os.scandir(edge_dir) if f.name.endswith('.jpg')])

    # Check if the number of files matches
    if len(input_files) != len(edge_files):
        print(f"INFO: Mismatch in number of images: {len(input_files)} input images vs {len(edge_files)} edge images.")
        print("INFO: Clearing edge folder and regenerating edge maps...")
        _clear_folder(edge_dir)
        _generate_edge_maps(split=split, batch_size=config.BATCH_SIZE_G1_INFERENCE)
        return False

    if len(input_files) == len(edge_files):
        # Check if all input images have corresponding edge maps
        for input_file in input_files:
            # Check for edge map file with expected naming pattern
            expected_edge_file = f"{os.path.splitext(input_file.name)[0]}_edge_map.jpg"
            if expected_edge_file not in edge_files:
                print(f"INFO: Missing edge map for {input_file.name}. Clearing edge folder and regenerating edge maps...")
                _clear_folder(edge_dir)
                _generate_edge_maps(split=split, batch_size=config.BATCH_SIZE_G1_INFERENCE)
                return False

    print("INFO: Number of images and corresponding edge maps match.")
    return True


def validate_guidance_images(split="train"):
    """
    Validates if guidance images exist for all input images.
    If any guidance image is missing, they will be generated.
    
    Guidance images combine both edge and color information for the G2 inpainting model.
    
    Args:
        split (str): Dataset split to use ('train', 'test', 'val', or 'demo').

    Returns:
        bool: True if validation successful (all guidance images exist), False if regeneration was needed.
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
    elif split == "demo":
        input_dir = config.DEMO_IMAGES_INPUT
        guidance_dir = config.DEMO_GUIDANCE_DIR
    else:
        raise ValueError("Invalid split. Choose from 'train', 'test', 'val', or 'demo'.")


    # Ensure guidance directory exists
    os.makedirs(guidance_dir, exist_ok=True)
    
    # Get list of input and guidance images
    input_path = Path(input_dir)
    guidance_path = Path(guidance_dir)
    
    # Use sets for faster membership testing (O(1) lookup)
    input_basenames = {os.path.splitext(f.name)[0] for f in input_path.glob("*.jpg")}
    guidance_basenames = {os.path.splitext(f.name)[0] for f in guidance_path.glob("*.jpg")}
    
    # First check: Count comparison
    if len(input_basenames) != len(guidance_basenames):
        print(f"INFO: Count mismatch: {len(input_basenames)} input images vs {len(guidance_basenames)} guidance images.")
        print("INFO: Generating guidance images...")
        _generate_guidance_images(split=split)
        return False
    
    # Second check: Name existence (ensure every input image has a guidance image)
    if not input_basenames.issubset(guidance_basenames):
        print("INFO: Some input images don't have corresponding guidance images.")
        print("INFO: Generating guidance images...")
        _generate_guidance_images(split=split)
        return False
    
    print("INFO: All guidance images exist.")
    return True


def _clear_folder(folder_path):
    """
    Clears all files in the specified folder.
    
    Used to remove outdated edge maps or guidance images before regeneration.
    
    Args:
        folder_path (str): Path to the folder to clear.
    """
    for file in os.scandir(folder_path):
        os.remove(file.path)
    print(f"INFO: Cleared folder: {folder_path}")


def _generate_edge_maps(split="train", batch_size=config.BATCH_SIZE_G1_INFERENCE):
    """
    Generates edge maps for all input images using the trained G1 model.
    
    Processes images in batches for efficiency, using the G1 edge generator to predict
    edges even in masked (missing) regions.
    
    Args:
        split (str): Dataset split to use ('train', 'test', 'val', or 'demo')
        batch_size (int): Batch size for processing images through the model
    """
    # Import here instead of at the top level to avoid circular imports
    from dataloader_g1 import get_dataloader_g1
    
    # Select input and edge directories based on the split
    if split == "train":
        edge_dir = config.TRAIN_EDGE_DIR
    elif split == "test":
        edge_dir = config.TEST_EDGE_DIR
    elif split == "val":
        edge_dir = config.VAL_EDGE_DIR
    elif split == "demo":
        edge_dir = config.DEMO_EDGE_DIR
    else:
        raise ValueError("Invalid split. Choose from 'train', 'test', 'val', or 'demo'.")


    # Ensure the edge directory exists
    os.makedirs(edge_dir, exist_ok=True)

    # Load the G1 model checkpoint
    checkpoint_path = config.G1_MODEL_PATH  # Path to the G1 model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

    # Initialize the model architecture
    model = EdgeGenerator()

    # Handle different checkpoint formats - some have wrapped state dicts, others are direct
    if "g1_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["g1_state_dict"])  # Load only the model weights
    else:
        model.load_state_dict(checkpoint)  # Directly load raw weights if no wrapper exists

    # Prepare model for inference
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
    model.to(config.DEVICE)  # Move the model to the specified device (e.g., GPU)

    # Initialize the dataloader with use_gt=False (we don't need ground truth) and filenames=True (to name saved files)
    dataloader = get_dataloader_g1(split=split, batch_size=config.BATCH_SIZE_G1_INFERENCE ,use_mask=True, use_gt=False, return_filenames=True)

    # Process images in batches
    for batch_idx, batch in enumerate(dataloader):
        # Get batch data and move to device
        input_edge = batch["input_edge"].to(config.DEVICE)  # Shape: (batch_size, 1, H, W)
        gray = batch["gray"].to(config.DEVICE)              # Shape: (batch_size, 1, H, W)
        mask = batch["mask"].to(config.DEVICE)              # Shape: (1, H, W)
        filenames = batch.get("filenames", None)            # Get filenames if available

        # Generate edge maps for the batch using the G1 model (without gradient calculation)
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

            # Save the edge map as a jpeg image
            cv2.imwrite(edge_path, edge_map)
        
        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"INFO: Processed batch {batch_idx + 1}/{len(dataloader)}")

    print(f"INFO: Generated edge maps for all input images in {edge_dir}.")
    
    # Clean up GPU memory to prevent leaks
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _process_single_guidance_image(args):
    """
    Process a single image for guidance generation.
    
    This function is designed to be used with multiprocessing for parallel guidance image generation.
    It takes a tuple of arguments to work with ProcessPoolExecutor.
    
    Args:
        args (tuple): Tuple containing (file, input_dir, edge_dir, guidance_dir)
            - file (Path): Path object for the input image file
            - input_dir (str): Directory containing input images
            - edge_dir (str): Directory containing edge maps
            - guidance_dir (str): Directory for saving generated guidance images
        
    Returns:
        bool: True if guidance image was generated successfully, False if skipped or failed
    """
    file, input_dir, edge_dir, guidance_dir = args
    basename = os.path.splitext(file.name)[0]
    input_path = str(file)  # Convert WindowsPath to string
    edge_path = os.path.join(edge_dir, f"{basename}_edge_map.jpg")
    guidance_path = os.path.join(guidance_dir, file.name)
    
    # Skip if guidance image already exists
    if os.path.exists(guidance_path):
        return False
        
    # Read input and edge images
    input_img = cv2.imread(input_path)  # RGB input image with masked areas
    edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)  # Generated edge map
    
    # Check for read errors
    if input_img is None or edge_img is None:
        print(f"WARNING: Could not read input or edge image for {file.name}")
        return False
    
    # Generate guidance image by combining inpainted color and edge information
    guidance_img = gen_guidance_img(input_img, edge_img)
    
    # Save the guidance image
    cv2.imwrite(guidance_path, guidance_img)
    return True


def _generate_guidance_images(split="train", num_workers=config.NUM_WORKERS):
    """
    Generates guidance images for all input images based on edge maps using parallel processing.
    
    Guidance images combine color information (from rough inpainting) with 
    structural information (from edge maps) to guide the G2 model.
    
    Args:
        split (str): Dataset split to use ('train', 'test', 'val', or 'demo').
        num_workers (int): Number of worker processes to use for parallel processing.
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
    elif split == "demo":
        input_dir = config.DEMO_IMAGES_INPUT
        guidance_dir = config.DEMO_GUIDANCE_DIR
        edge_dir = config.DEMO_EDGE_DIR
    else:
        raise ValueError("Invalid split. Choose from 'train', 'test', 'val', or 'demo'.")

    # First make sure edge maps exist - generate them if needed
    validate_edge_map(split)
    print(f"INFO: Edge maps validated for {split} split.")
    
    # Ensure guidance directory exists
    os.makedirs(guidance_dir, exist_ok=True)
    
    # Get all input images
    input_path = Path(input_dir)
    input_files = sorted([f for f in input_path.glob("*.jpg")])
    total_files = len(input_files)
    
    print(f"INFO: Generating guidance images for {total_files} input images using {num_workers} workers...")
    
    # Prepare argument tuples for each task to pass to the worker function
    tasks = [(file, input_dir, edge_dir, guidance_dir) for file in input_files]
    
    # Process images in parallel using a process pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Track progress with counters
        completed = 0
        failed = 0
        
        # Process all tasks with progress tracking via tqdm
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
    
    print(f"INFO: Generated {completed} guidance images in {guidance_dir}. Failed: {failed}.")