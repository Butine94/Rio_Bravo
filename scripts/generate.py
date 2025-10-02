
import torch
import yaml
import os
from pathlib import Path

from models.diffusion import DiffusionGenerator
from utils.text_processing import parse_script, extract_scenes
from utils.video_tools import create_video
from utils.file_helpers import setup_directories, save_metadata


def load_configuration(config_path='config.yaml'):
    """Load settings from YAML configuration file"""
    if not Path(config_path).exists():
        print(f"Configuration file not found: {config_path}")
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded configuration from {config_path}")
    return config


def get_default_config():
    """Return default configuration"""
    return {
        'model_name': 'runwayml/stable-diffusion-v1-5',
        'output_path': 'outputs',
        'frames_path': 'outputs/frames',
        'resolution': {
            'width': 768,
            'height': 512
        },
        'inference_steps': 20,
        'guidance_scale': 7.5,
        'random_seed': 42,
        'num_scenes': 4,
        'video_fps': 24,
        'scene_duration': 3.0
    }


def run_generation():
    """Execute the film generation pipeline"""
    print("=" * 60)
    print("CineAI Film Generation System")
    print("=" * 60)
    
    # Load configuration
    config = load_configuration()
    
    # Setup output directories
    output_dir = Path(config['output_path'])
    frames_dir = Path(config['frames_path'])
    setup_directories([output_dir, frames_dir])
    
    # Load input script
    script_file = 'scripts/input.txt'
    print(f"\nLoading script from {script_file}")
    
    if not Path(script_file).exists():
        print(f"Script file not found, using default content")
        script_text = get_default_script()
    else:
        with open(script_file, 'r') as f:
            script_text = f.read()
    
    print(f"Script loaded: {len(script_text)} characters")
    
    # Parse script into scenes
    print(f"\nProcessing script into {config['num_scenes']} scenes")
    scenes = parse_script(script_text, num_scenes=config['num_scenes'])
    
    for i, scene in enumerate(scenes, 1):
        print(f"  Scene {i}: {scene['shot_type']}")
    
    # Initialize diffusion model
    print(f"\nInitializing model: {config['model_name']}")
    generator = DiffusionGenerator(config)
    
    # Generate images
    print("\nGenerating images")
    image_files = []
    
    for i, scene in enumerate(scenes):
        print(f"  Generating scene {i+1}/{len(scenes)}: {scene['shot_type']}")
        
        output_path = frames_dir / f"frame_{i+1:03d}.png"
        generator.generate_image(
            prompt=scene['prompt'],
            output_file=str(output_path)
        )
        
        image_files.append(str(output_path))
        print(f"    Saved to {output_path.name}")
    
    # Create video
    video_output = output_dir / 'output.mp4'
    print(f"\nCompiling video to {video_output}")
    
    success = create_video(
        image_paths=image_files,
        output_path=str(video_output),
        fps=config['video_fps'],
        duration=config['scene_duration']
    )
    
    if success:
        print(f"Video created successfully: {video_output}")
    else:
        print("Video creation failed or skipped")
    
    # Save metadata
    metadata = {
        'scenes': scenes,
        'config': config,
        'output_files': image_files
    }
    save_metadata(metadata, str(output_dir / 'metadata.json'))
    
    # Cleanup
    generator.cleanup()
    
    print("\n" + "=" * 60)
    print("Generation complete")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


def get_default_script():
    """Return default script content"""
    return """
    The vast desert stretches endlessly beneath a crimson sunset sky.
    Two cargo trains approach each other on parallel iron tracks.
    Weathered rock formations tower above the railway corridor.
    The locomotives thunder past in opposite directions, dust rising.
    Silence returns as the trains vanish into the twilight distance.
    """


if __name__ == "__main__":
    run_generation()