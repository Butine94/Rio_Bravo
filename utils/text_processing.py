"""
Text processing utilities for script parsing
"""

import re


# Shot type definitions
SHOT_TYPES = [
    'wide establishing shot',
    'medium composition',
    'close detail shot',
    'overhead perspective',
    'low angle view',
    'tracking shot'
]


def parse_script(text, num_scenes=4):
    """
    Parse script text into structured scene descriptions
    
    Args:
        text: Raw script text
        num_scenes: Number of scenes to generate
        
    Returns:
        List of scene dictionaries
    """
    # Clean and split text
    sentences = split_into_segments(text)
    
    if not sentences:
        print("Warning: No valid content found, using fallback scenes")
        return generate_fallback_scenes(num_scenes)
    
    # Extract visual descriptions
    visual_elements = extract_visual_elements(sentences)
    
    # Build scene list
    scenes = []
    for i in range(num_scenes):
        shot_type = SHOT_TYPES[i % len(SHOT_TYPES)]
        
        if visual_elements:
            element_idx = i % len(visual_elements)
            description = f"{shot_type} of {visual_elements[element_idx]}"
        else:
            description = get_default_description(i)
        
        scenes.append({
            'id': i + 1,
            'shot_type': shot_type,
            'prompt': description
        })
    
    return scenes


def extract_scenes(text):
    """
    Extract distinct scenes from script text
    
    Args:
        text: Script text
        
    Returns:
        List of scene strings
    """
    return split_into_segments(text)


def split_into_segments(text):
    """
    Split text into sentence segments
    
    Args:
        text: Input text
        
    Returns:
        List of sentence strings
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split on sentence boundaries
    segments = re.split(r'[.!?]+', text)
    
    # Filter short segments
    segments = [s.strip() for s in segments if len(s.strip()) > 20]
    
    return segments


def extract_visual_elements(segments):
    """
    Extract visual descriptions from text segments
    
    Args:
        segments: List of text segments
        
    Returns:
        List of visual description strings
    """
    visuals = []
    
    # Define visual keyword categories
    visual_keywords = {
        'environment': ['desert', 'ocean', 'mountain', 'forest', 'city', 'space', 'valley'],
        'objects': ['train', 'building', 'vehicle', 'structure', 'formation', 'machine'],
        'subjects': ['figure', 'silhouette', 'person', 'character', 'traveler'],
        'lighting': ['sunset', 'dawn', 'twilight', 'shadow', 'light', 'darkness', 'glow']
    }
    
    for segment in segments:
        segment_lower = segment.lower()
        
        # Check for presence of visual keywords
        has_visuals = False
        for category, keywords in visual_keywords.items():
            if any(keyword in segment_lower for keyword in keywords):
                has_visuals = True
                break
        
        if has_visuals:
            # Use first clause of segment
            parts = segment.split(',')
            visual_desc = parts[0].strip()
            visuals.append(visual_desc)
    
    return visuals[:10]


def get_default_description(index):
    """
    Get fallback description when parsing fails
    
    Args:
        index: Scene index
        
    Returns:
        Default description string
    """
    defaults = [
        "expansive natural landscape with atmospheric conditions",
        "architectural elements with geometric patterns",
        "textured surfaces with dramatic lighting",
        "silhouetted forms against gradient sky",
        "environmental details with depth variation",
        "structural composition with leading lines"
    ]
    
    return defaults[index % len(defaults)]


def generate_fallback_scenes(num_scenes):
    """
    Generate default scenes when script parsing fails
    
    Args:
        num_scenes: Number of scenes to generate
        
    Returns:
        List of scene dictionaries
    """
    scenes = []
    
    for i in range(num_scenes):
        scenes.append({
            'id': i + 1,
            'shot_type': SHOT_TYPES[i % len(SHOT_TYPES)],
            'prompt': get_default_description(i)
        })
    
    return scenes