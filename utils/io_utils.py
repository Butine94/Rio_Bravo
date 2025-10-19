"""
File I/O and video creation utilities
Essential operations only - zero quality loss
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional


def setup_directories(dir_list: List[str]) -> None:
    """Create required directories if they don't exist"""
    for directory in dir_list:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"Directories ready: {', '.join(dir_list)}")


def read_text_file(filepath: str) -> str:
    """Read text file with error handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return ""
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""


def save_metadata(data: Dict, filepath: str) -> bool:
    """Save metadata to JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Metadata saved: {filepath}")
        return True
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return False


def load_metadata(filepath: str) -> Optional[Dict]:
    """Load metadata from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Metadata file not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def list_image_files(directory: str, extension: str = '.png') -> List[str]:
    """List all image files in directory"""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    return sorted([str(f) for f in dir_path.glob(f'*{extension}')])


def create_video(image_paths: List[str], output_path: str, 
                fps: int = 24, duration: float = 3.0) -> bool:
    """
    Compile images into video with professional quality settings
    Maintains full image quality, proper codec settings
    """
    try:
        from moviepy.editor import ImageClip, concatenate_videoclips
    except ImportError:
        print("MoviePy not installed. Install with: pip install moviepy")
        return False
    
    valid_images = [p for p in image_paths if Path(p).exists()]
    
    if not valid_images:
        print("Error: No valid images found for video compilation")
        return False
    
    print(f"Compiling {len(valid_images)} images into video")
    
    try:
        clips = [ImageClip(img).set_duration(duration) for img in valid_images]
        video = concatenate_videoclips(clips, method="compose")
        
        video.write_videofile(
            output_path,
            fps=fps,
            codec='libx264',
            audio=False,
            verbose=False,
            logger=None,
            bitrate="8000k"  # High bitrate for quality
        )
        
        video.close()
        for clip in clips:
            clip.close()
        
        print(f"Video created: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating video: {e}")
        return False


def create_video_with_transitions(image_paths: List[str], output_path: str,
                                  transition_duration: float = 0.5, 
                                  shot_duration: float = 3.0,
                                  fps: int = 24) -> bool:
    """
    Create video with crossfade transitions
    Professional quality with smooth transitions
    """
    try:
        from moviepy.editor import ImageClip, CompositeVideoClip
    except ImportError:
        print("MoviePy not available for transitions")
        return False
    
    try:
        clips = []
        current_time = 0
        
        for i, img_path in enumerate(image_paths):
            if not Path(img_path).exists():
                continue
            
            clip = ImageClip(img_path).set_duration(shot_duration)
            
            if i > 0:
                clip = clip.crossfadein(transition_duration)
                clip = clip.set_start(current_time - transition_duration)
                current_time += (shot_duration - transition_duration)
            else:
                clip = clip.set_start(current_time)
                current_time += shot_duration
            
            clips.append(clip)
        
        final_video = CompositeVideoClip(clips)
        final_video.write_videofile(
            output_path,
            fps=fps,
            codec='libx264',
            verbose=False,
            logger=None,
            bitrate="8000k"  # High bitrate for quality
        )
        
        final_video.close()
        for clip in clips:
            clip.close()
        
        print(f"Video with transitions created: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating transitions: {e}")
        return False