"""
Text processing for script parsing and scene extraction
Core parsing logic - zero quality loss
"""

import re
from typing import List, Dict

SHOT_TYPES = [
    'wide establishing shot',
    'medium composition',
    'close detail shot',
    'overhead perspective',
    'low angle view',
    'tracking shot'
]

VISUAL_KEYWORDS = {
    'environment': ['desert', 'ocean', 'mountain', 'forest', 'city', 'space', 'valley'],
    'objects': ['train', 'building', 'vehicle', 'structure', 'formation', 'machine'],
    'subjects': ['figure', 'silhouette', 'person', 'character', 'traveler'],
    'lighting': ['sunset', 'dawn', 'twilight', 'shadow', 'light', 'darkness', 'glow']
}

DEFAULT_DESCRIPTIONS = [
    "expansive natural landscape with atmospheric conditions",
    "architectural elements with geometric patterns",
    "textured surfaces with dramatic lighting",
    "silhouetted forms against gradient sky",
    "environmental details with depth variation",
    "structural composition with leading lines"
]


def parse_script(text: str, num_scenes: int = 4) -> List[Dict]:
    """
    Parse script text into structured scene descriptions
    Full-quality parsing with visual element extraction
    """
    sentences = split_into_segments(text)
    
    if not sentences:
        print("Warning: No valid content found, using fallback scenes")
        return generate_fallback_scenes(num_scenes)
    
    visual_elements = extract_visual_elements(sentences)
    
    scenes = []
    for i in range(num_scenes):
        shot_type = SHOT_TYPES[i % len(SHOT_TYPES)]
        
        if visual_elements:
            element_idx = i % len(visual_elements)
            description = f"{shot_type} of {visual_elements[element_idx]}"
        else:
            description = DEFAULT_DESCRIPTIONS[i % len(DEFAULT_DESCRIPTIONS)]
        
        scenes.append({
            'id': i + 1,
            'shot_type': shot_type,
            'prompt': description
        })
    
    return scenes


def split_into_segments(text: str) -> List[str]:
    """Split text into meaningful sentence segments"""
    text = re.sub(r'\s+', ' ', text).strip()
    segments = re.split(r'[.!?]+', text)
    segments = [s.strip() for s in segments if len(s.strip()) > 20]
    return segments


def extract_visual_elements(segments: List[str]) -> List[str]:
    """
    Extract visual descriptions from text segments
    Preserves all visual keywords for maximum quality
    """
    visuals = []
    
    for segment in segments:
        segment_lower = segment.lower()
        
        has_visuals = any(
            keyword in segment_lower
            for keywords in VISUAL_KEYWORDS.values()
            for keyword in keywords
        )
        
        if has_visuals:
            parts = segment.split(',')
            visual_desc = parts[0].strip()
            visuals.append(visual_desc)
    
    return visuals[:10]


def generate_fallback_scenes(num_scenes: int) -> List[Dict]:
    """Generate high-quality default scenes when parsing fails"""
    return [
        {
            'id': i + 1,
            'shot_type': SHOT_TYPES[i % len(SHOT_TYPES)],
            'prompt': DEFAULT_DESCRIPTIONS[i % len(DEFAULT_DESCRIPTIONS)]
        }
        for i in range(num_scenes)
    ]