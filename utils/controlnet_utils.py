"""
ControlNet preprocessing utilities
Core functions for depth maps and composition control
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, List
import os


class ControlNetPreprocessor:
    """Preprocessing utilities for ControlNet conditioning"""
    
    @staticmethod
    def create_depth_map(width: int = 512, height: int = 512, 
                        gradient_type: str = "vertical") -> Image.Image:
        """
        Create depth map for spatial composition control
        
        Args:
            width: Output width
            height: Output height
            gradient_type: "vertical", "horizontal", "radial", or "center"
        
        Returns:
            PIL Image depth map
        """
        if gradient_type == "vertical":
            gray = np.linspace(255, 128, height, dtype=np.uint8)
            gray = np.tile(gray.reshape(-1, 1), (1, width))
        
        elif gradient_type == "horizontal":
            gray = np.linspace(255, 128, width, dtype=np.uint8)
            gray = np.tile(gray.reshape(1, -1), (height, 1))
        
        elif gradient_type == "radial":
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            gray = (255 - (dist / max_dist * 127)).astype(np.uint8)
        
        elif gradient_type == "center":
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2
            dist_y = np.abs(y - center_y) / (height / 2)
            dist_x = np.abs(x - center_x) / (width / 2)
            dist = np.maximum(dist_y, dist_x)
            gray = (255 - (dist * 100)).clip(100, 255).astype(np.uint8)
        
        else:
            gray = np.linspace(255, 128, height, dtype=np.uint8)
            gray = np.tile(gray.reshape(-1, 1), (1, width))
        
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        depth_image = Image.fromarray(gray).convert("RGB")
        return depth_image
    
    @staticmethod
    def create_depth_from_image(image_path: str, width: int = 512, 
                               height: int = 512) -> Image.Image:
        """
        Convert existing image to depth map
        
        Args:
            image_path: Path to reference image
            width: Output width
            height: Output height
        
        Returns:
            PIL Image depth map
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (width, height))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        depth = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        depth_image = Image.fromarray(depth).convert("RGB")
        return depth_image
    
    @staticmethod
    def create_shot_sequence_depths(num_shots: int, shot_types: List[str],
                                   width: int = 768, height: int = 512) -> List[Image.Image]:
        """
        Create optimized depth maps for a shot sequence
        
        Args:
            num_shots: Number of shots
            shot_types: List of shot types (establishing, medium, closeup)
            width: Output width
            height: Output height
        
        Returns:
            List of PIL Image depth maps
        """
        depth_maps = []
        
        for shot_type in shot_types[:num_shots]:
            shot_lower = shot_type.lower()
            
            if any(t in shot_lower for t in ["establishing", "wide", "master"]):
                gradient_type = "vertical"
            elif any(t in shot_lower for t in ["medium", "mid"]):
                gradient_type = "center"
            elif any(t in shot_lower for t in ["closeup", "close-up", "cu"]):
                gradient_type = "radial"
            else:
                gradient_type = "vertical"
            
            depth = ControlNetPreprocessor.create_depth_map(width, height, gradient_type)
            depth_maps.append(depth)
        
        return depth_maps


def demo_preprocessor():
    """Demonstrate preprocessing capabilities"""
    print("ControlNet Preprocessor Demo\n")
    
    processor = ControlNetPreprocessor()
    os.makedirs("./outputs/depth_maps", exist_ok=True)
    
    depth_types = ["vertical", "horizontal", "radial", "center"]
    
    for depth_type in depth_types:
        print(f"Creating {depth_type} depth map...")
        depth = processor.create_depth_map(768, 512, gradient_type=depth_type)
        depth.save(f"./outputs/depth_maps/depth_{depth_type}.png")
    
    print("\nDemo complete! Check ./outputs/depth_maps/")


if __name__ == "__main__":
    demo_preprocessor()