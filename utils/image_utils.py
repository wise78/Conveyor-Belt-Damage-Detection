#!/usr/bin/env python3
"""
Image processing utilities for conveyor belt damage detection system
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Optional, Dict, Any


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from path with error handling
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image as numpy array or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> bool:
    """
    Save image with specified quality
    
    Args:
        image: Image to save
        output_path: Output file path
        quality: JPEG quality (1-100)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return True
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                keep_aspect: bool = True) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: Target (width, height)
        keep_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if keep_aspect:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create canvas with target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    else:
        return cv2.resize(image, target_size)


def apply_roi_mask(image: np.ndarray, roi_coords: List[int], 
                  roi_shape: str = "rectangle") -> np.ndarray:
    """
    Apply ROI mask to image
    
    Args:
        image: Input image
        roi_coords: ROI coordinates
        roi_shape: Shape type ("rectangle", "polygon", "circle")
        
    Returns:
        Image with ROI mask applied
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    if roi_shape == "rectangle":
        x, y, w, h = roi_coords
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    elif roi_shape == "polygon":
        points = np.array(roi_coords, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    elif roi_shape == "circle":
        center = (roi_coords[0], roi_coords[1])
        radius = roi_coords[2]
        cv2.circle(mask, center, radius, 255, -1)
    
    # Apply mask
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image


def crop_roi(image: np.ndarray, roi_coords: List[int], 
            roi_shape: str = "rectangle") -> np.ndarray:
    """
    Crop image to ROI area
    
    Args:
        image: Input image
        roi_coords: ROI coordinates
        roi_shape: Shape type ("rectangle", "polygon", "circle")
        
    Returns:
        Cropped image
    """
    if roi_shape == "rectangle":
        x, y, w, h = roi_coords
        return image[y:y+h, x:x+w]
    elif roi_shape == "polygon":
        points = np.array(roi_coords, dtype=np.int32)
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(points)
        return image[y:y+h, x:x+w]
    elif roi_shape == "circle":
        center_x, center_y, radius = roi_coords
        x1, y1 = max(0, center_x - radius), max(0, center_y - radius)
        x2, y2 = min(image.shape[1], center_x + radius), min(image.shape[0], center_y + radius)
        return image[y1:y2, x1:x2]
    
    return image


def calculate_image_quality(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate image quality metrics
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with quality metrics
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian variance (sharpness)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate brightness
    brightness = np.mean(gray)
    
    # Calculate contrast
    contrast = np.std(gray)
    
    return {
        "laplacian_variance": laplacian_var,
        "brightness": brightness,
        "contrast": contrast
    }


def is_image_quality_good(image: np.ndarray, 
                         min_lap_var: float = 60.0,
                         min_bright: float = 25.0,
                         max_bright: float = 235.0) -> bool:
    """
    Check if image quality meets minimum requirements
    
    Args:
        image: Input image
        min_lap_var: Minimum Laplacian variance
        min_bright: Minimum brightness
        max_bright: Maximum brightness
        
    Returns:
        True if quality is good, False otherwise
    """
    quality = calculate_image_quality(image)
    
    return (quality["laplacian_variance"] >= min_lap_var and
            min_bright <= quality["brightness"] <= max_bright)


def draw_detection_results(image: np.ndarray, 
                          detections: List[Dict[str, Any]],
                          class_names: List[str] = None,
                          colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Draw detection results on image
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        class_names: List of class names
        colors: List of colors for each class
        
    Returns:
        Image with detections drawn
    """
    result_image = image.copy()
    
    if colors is None:
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for detection in detections:
        # Extract detection info
        bbox = detection.get("bbox", [])
        confidence = detection.get("confidence", 0.0)
        class_id = detection.get("class_id", 0)
        
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if class_names and class_id < len(class_names):
                label = f"{class_names[class_id]}: {confidence:.2f}"
            else:
                label = f"Class {class_id}: {confidence:.2f}"
            
            # Calculate text size
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw text background
            cv2.rectangle(result_image, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result_image


def create_video_writer(output_path: str, fps: int = 30, 
                       frame_size: Tuple[int, int] = (640, 480)) -> cv2.VideoWriter:
    """
    Create video writer for saving output video
    
    Args:
        output_path: Output video path
        fps: Frames per second
        frame_size: Frame size (width, height)
        
    Returns:
        VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to config file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")
        return False