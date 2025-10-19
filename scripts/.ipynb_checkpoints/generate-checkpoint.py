#!/usr/bin/env python3
"""
Rio Bravo - Production Pipeline
Screenplay to cinematic storyboards with ControlNet and LoRA
"""

import yaml
import argparse
from pathlib import Path
from models.diffusion import CinematicDiffusionModel
from utils.text_processing import parse_script
from utils.io_utils import setup_directories, read_text_file, save_metadata, create_video

def main():
    """Main pipeline execution"""
    
    parser = argparse.ArgumentParser(description='Generate cinematic storyboards from screenplay')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--script', default=None, help='Script file path (overrides config)')
    parser.add_argument('--output', default=None, help='Output directory (overrides config)')
    parser.add_argument('--num-shots', type=int, default=None, help='Number of shots')
    parser.add_argument('--video', action='store_true', help='Create video compilation')
    parser.add_argument('--depth-consistency', action='store_true', help='Use depth consistency')
    args = parser.parse_args()
    
    print("Loading configuration...")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    script_path = args.script if args.script else config['script']['screenplay_path']
    output_dir = args.output if args.output else config['output']['directory']
    num_shots = args.num_shots if args.num_shots else config['scene']['max_shots']
    
    setup_directories([output_dir])
    
    print(f"Reading script from {script_path}...")
    script_text = read_text_file(script_path)
    
    if not script_text:
        print("Error: Could not read script file")
        return
    
    print(f"Parsing script into {num_shots} shots...")
    scenes = parse_script(script_text, num_scenes=num_shots)
    
    print(f"\nGenerated {len(scenes)} scenes:")
    for scene in scenes:
        print(f"  Shot {scene['id']}: {scene['shot_type']}")
        print(f"    {scene['prompt'][:80]}...")
    
    print("\nInitializing diffusion model...")
    model = CinematicDiffusionModel(config)
    
    print(f"\nGenerating {len(scenes)} cinematic shots...")
    print(f"  Resolution: {config['diffusion']['width']}x{config['diffusion']['height']}")
    print(f"  Steps: {config['diffusion']['num_inference_steps']}")
    print(f"  ControlNet: {'Enabled' if config['diffusion']['use_controlnet'] else 'Disabled'}")
    print(f"  LoRA: {'Enabled' if config['diffusion']['use_lora'] else 'Disabled'}")
    print(f"  Depth Consistency: {'Enabled' if args.depth_consistency else 'Disabled'}\n")
    
    generated_scenes = model.generate_shots(
        scenes, 
        output_dir,
        use_depth_consistency=args.depth_consistency
    )
    
    metadata = {
        'config': config,
        'script_path': script_path,
        'num_shots': len(generated_scenes),
        'scenes': generated_scenes
    }
    save_metadata(metadata, str(Path(output_dir) / 'metadata.json'))
    
    print(f"\nGeneration complete!")
    print(f"  Output: {output_dir}")
    print(f"  Shots: {len(generated_scenes)}")
    
    if args.video:
        print("\nCreating video...")
        image_paths = [scene['image_path'] for scene in generated_scenes]
        video_path = Path(output_dir) / 'rio_bravo_sequence.mp4'
        
        if create_video(image_paths, str(video_path)):
            print(f"Video: {video_path}")
        else:
            print("Video creation failed (requires moviepy)")
    
    model.cleanup()
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()