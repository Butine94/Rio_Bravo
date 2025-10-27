"""
File system operations and utilities
"""

import os
import json
from pathlib import Path


def setup_directories(dir_list):
    """
    Create required directories if they don't exist
    
    Args:
        dir_list: List of directory paths to create
    """
    for directory in dir_list:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directories ready: {', '.join([str(d) for d in dir_list])}")


def save_metadata(data, filepath):
    """
    Save metadata to JSON file
    
    Args:
        data: Dictionary of metadata
        filepath: Output file path
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Metadata saved to {filepath}")
    except Exception as e:
        print(f"Error saving metadata: {e}")


def load_metadata(filepath):
    """
    Load metadata from JSON file
    
    Args:
        filepath: Path to metadata file
        
    Returns:
        Dictionary of metadata or None if failed
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Metadata file not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def read_text_file(filepath):
    """
    Read text file with error handling
    
    Args:
        filepath: Path to text file
        
    Returns:
        File contents as string
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return ""
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""


def list_image_files(directory, extension='.png'):
    """
    List all image files in directory
    
    Args:
        directory: Directory path
        extension: File extension to filter
        
    Returns:
        List of image file paths
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return []
    
    image_files = sorted([
        str(f) for f in dir_path.glob(f'*{extension}')
    ])
    
    return image_files


def validate_file_exists(filepath):
    """
    Check if file exists
    
    Args:
        filepath: Path to check
        
    Returns:
        Boolean indicating existence
    """
    return Path(filepath).exists()


def get_file_size(filepath):
    """
    Get file size in bytes
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in bytes or 0 if not found
    """
    try:
        return Path(filepath).stat().st_size
    except Exception:
        return 0