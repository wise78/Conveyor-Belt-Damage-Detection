#!/usr/bin/env python3
"""
ROI (Region of Interest) utilities for conveyor belt damage detection system
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


class ROIManager:
    """
    Manager class for handling ROI operations
    """
    
    def __init__(self, mode: str = "manual", coords: List[int] = None, 
                 shape: str = "rectangle"):
        """
        Initialize ROI Manager
        
        Args:
            mode: ROI mode ("manual", "auto", "interactive")
            coords: ROI coordinates
            shape: ROI shape ("rectangle", "polygon", "circle")
        """
        self.mode = mode
        self.shape = shape
        self.coords = coords or [100, 100, 500, 400]  # Default rectangle
        self.polygon_points = []  # For polygon mode
        self.circle_center = (300, 250)  # For circle mode
        self.circle_radius = 150  # For circle mode
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.roi_set = False
        self.polygon_drawing = False
        
    def set_roi_from_frame(self, frame: np.ndarray) -> None:
        """
        Set ROI based on frame dimensions
        
        Args:
            frame: Input frame
        """
        h, w = frame.shape[:2]
        
        if self.mode == "auto":
            # Full frame ROI
            if self.shape == "rectangle":
                self.coords = [0, 0, w, h]
            elif self.shape == "circle":
                self.circle_center = (w//2, h//2)
                self.circle_radius = min(w, h) // 3
            elif self.shape == "polygon":
                # Create default polygon (square)
                margin = min(w, h) // 4
                self.polygon_points = [
                    [margin, margin],
                    [w-margin, margin],
                    [w-margin, h-margin],
                    [margin, h-margin]
                ]
            self.roi_set = True
            
        elif self.mode == "manual":
            # Manual ROI with default 50% of frame
            if self.shape == "rectangle":
                x = w // 4
                y = h // 4
                width = w // 2
                height = h // 2
                self.coords = [x, y, width, height]
            elif self.shape == "circle":
                self.circle_center = (w//2, h//2)
                self.circle_radius = min(w, h) // 4
            elif self.shape == "polygon":
                margin = min(w, h) // 4
                self.polygon_points = [
                    [margin, margin],
                    [w-margin, margin],
                    [w-margin, h-margin],
                    [margin, h-margin]
                ]
            self.roi_set = True
    
    def get_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Get ROI mask for given frame
        
        Args:
            frame: Input frame
            
        Returns:
            ROI mask
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        if self.shape == "rectangle":
            x, y, w, h = self.coords
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        elif self.shape == "polygon":
            if self.polygon_points:
                points = np.array(self.polygon_points, dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
        elif self.shape == "circle":
            cv2.circle(mask, self.circle_center, self.circle_radius, 255, -1)
        
        return mask
    
    def apply_roi_to_frame(self, frame: np.ndarray, apply_mode: str = "mask") -> np.ndarray:
        """
        Apply ROI to frame
        
        Args:
            frame: Input frame
            apply_mode: Application mode ("mask", "crop")
            
        Returns:
            Processed frame
        """
        if apply_mode == "mask":
            mask = self.get_roi_mask(frame)
            return cv2.bitwise_and(frame, frame, mask=mask)
        elif apply_mode == "crop":
            return self.crop_roi_from_frame(frame)
        
        return frame
    
    def crop_roi_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Crop ROI from frame
        
        Args:
            frame: Input frame
            
        Returns:
            Cropped frame
        """
        if self.shape == "rectangle":
            x, y, w, h = self.coords
            return frame[y:y+h, x:x+w]
        elif self.shape == "polygon":
            if self.polygon_points:
                points = np.array(self.polygon_points, dtype=np.int32)
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(points)
                return frame[y:y+h, x:x+w]
        elif self.shape == "circle":
            center_x, center_y = self.circle_center
            radius = self.circle_radius
            x1, y1 = max(0, center_x - radius), max(0, center_y - radius)
            x2, y2 = min(frame.shape[1], center_x + radius), min(frame.shape[0], center_y + radius)
            return frame[y1:y2, x1:x2]
        
        return frame
    
    def draw_roi_on_frame(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255),
                         thickness: int = 2) -> np.ndarray:
        """
        Draw ROI on frame
        
        Args:
            frame: Input frame
            color: ROI color (B, G, R)
            thickness: Line thickness
            
        Returns:
            Frame with ROI drawn
        """
        result_frame = frame.copy()
        
        if self.shape == "rectangle":
            x, y, w, h = self.coords
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, thickness)
            
        elif self.shape == "polygon":
            if self.polygon_points:
                points = np.array(self.polygon_points, dtype=np.int32)
                cv2.polylines(result_frame, [points], True, color, thickness)
                
        elif self.shape == "circle":
            cv2.circle(result_frame, self.circle_center, self.circle_radius, color, thickness)
        
        return result_frame
    
    def save_roi_config(self, config_path: str) -> bool:
        """
        Save ROI configuration to file
        
        Args:
            config_path: Path to config file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = {
                "mode": self.mode,
                "shape": self.shape,
                "coords": self.coords,
                "polygon_points": self.polygon_points,
                "circle_center": self.circle_center,
                "circle_radius": self.circle_radius,
                "roi_set": self.roi_set
            }
            
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
            
        except Exception as e:
            print(f"Error saving ROI config to {config_path}: {e}")
            return False
    
    def load_roi_config(self, config_path: str) -> bool:
        """
        Load ROI configuration from file
        
        Args:
            config_path: Path to config file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.mode = config.get("mode", self.mode)
            self.shape = config.get("shape", self.shape)
            self.coords = config.get("coords", self.coords)
            self.polygon_points = config.get("polygon_points", self.polygon_points)
            self.circle_center = config.get("circle_center", self.circle_center)
            self.circle_radius = config.get("circle_radius", self.circle_radius)
            self.roi_set = config.get("roi_set", self.roi_set)
            
            return True
            
        except Exception as e:
            print(f"Error loading ROI config from {config_path}: {e}")
            return False


def create_interactive_roi(frame: np.ndarray, window_name: str = "Set ROI") -> Optional[ROIManager]:
    """
    Create interactive ROI selection
    
    Args:
        frame: Input frame
        window_name: Window name for display
        
    Returns:
        ROIManager object or None if cancelled
    """
    roi_manager = ROIManager(mode="interactive")
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_manager.start_point = (x, y)
            roi_manager.drawing = True
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if roi_manager.drawing:
                roi_manager.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            roi_manager.drawing = False
            roi_manager.end_point = (x, y)
            
            # Set rectangle coordinates
            x1, y1 = roi_manager.start_point
            x2, y2 = roi_manager.end_point
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            
            roi_manager.coords = [x, y, w, h]
            roi_manager.shape = "rectangle"
            roi_manager.roi_set = True
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        display_frame = roi_manager.draw_roi_on_frame(frame)
        
        if roi_manager.drawing and roi_manager.end_point:
            # Draw temporary rectangle
            x1, y1 = roi_manager.start_point
            x2, y2 = roi_manager.end_point
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(display_frame, "Click and drag to set ROI", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'S' to save, 'C' to cancel", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and roi_manager.roi_set:
            break
        elif key == ord('c'):
            roi_manager = None
            break
        elif key == 27:  # ESC
            roi_manager = None
            break
    
    cv2.destroyWindow(window_name)
    return roi_manager


def validate_roi_coordinates(coords: List[int], frame_shape: Tuple[int, int], 
                           shape: str = "rectangle") -> bool:
    """
    Validate ROI coordinates
    
    Args:
        coords: ROI coordinates
        frame_shape: Frame shape (height, width)
        shape: ROI shape
        
    Returns:
        True if valid, False otherwise
    """
    h, w = frame_shape
    
    if shape == "rectangle":
        if len(coords) != 4:
            return False
        x, y, width, height = coords
        return (0 <= x < w and 0 <= y < h and 
                x + width <= w and y + height <= h and
                width > 0 and height > 0)
                
    elif shape == "polygon":
        if len(coords) < 6 or len(coords) % 2 != 0:
            return False
        for i in range(0, len(coords), 2):
            x, y = coords[i], coords[i+1]
            if not (0 <= x < w and 0 <= y < h):
                return False
        return True
        
    elif shape == "circle":
        if len(coords) != 3:
            return False
        center_x, center_y, radius = coords
        return (0 <= center_x < w and 0 <= center_y < h and
                radius > 0 and
                center_x - radius >= 0 and center_x + radius <= w and
                center_y - radius >= 0 and center_y + radius <= h)
    
    return False


def get_roi_area(coords: List[int], shape: str = "rectangle") -> float:
    """
    Calculate ROI area
    
    Args:
        coords: ROI coordinates
        shape: ROI shape
        
    Returns:
        ROI area in pixels
    """
    if shape == "rectangle":
        if len(coords) == 4:
            width, height = coords[2], coords[3]
            return width * height
            
    elif shape == "polygon":
        if len(coords) >= 6 and len(coords) % 2 == 0:
            points = []
            for i in range(0, len(coords), 2):
                points.append([coords[i], coords[i+1]])
            points = np.array(points, dtype=np.float32)
            return cv2.contourArea(points)
            
    elif shape == "circle":
        if len(coords) == 3:
            radius = coords[2]
            return np.pi * radius * radius
    
    return 0.0