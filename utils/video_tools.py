"""
Video creation and manipulation utilities
"""

from pathlib import Path


def create_video(image_paths, output_path, fps=24, duration=3.0):
    """
    Compile sequence of images into video file
    
    Args:
        image_paths: List of image file paths
        output_path: Output video file path
        fps: Frames per second
        duration: Duration per image in seconds
        
    Returns:
        Boolean indicating success
    """
    try:
        from moviepy.editor import ImageClip, concatenate_videoclips
    except ImportError:
        print("MoviePy not installed, skipping video creation")
        print("Install with: pip install moviepy")
        return False
    
    # Validate image files exist
    valid_images = []
    for path in image_paths:
        if Path(path).exists():
            valid_images.append(path)
        else:
            print(f"Warning: Image not found: {path}")
    
    if not valid_images:
        print("Error: No valid images found for video compilation")
        return False
    
    print(f"Compiling {len(valid_images)} images into video")
    
    try:
        # Create clips from images
        clips = []
        for img_path in valid_images:
            clip = ImageClip(img_path).set_duration(duration)
            clips.append(clip)
        
        # Concatenate clips
        video = concatenate_videoclips(clips, method="compose")
        
        # Write output file
        video.write_videofile(
            output_path,
            fps=fps,
            codec='libx264',
            audio=False,
            verbose=False,
            logger=None
        )
        
        # Cleanup
        video.close()
        for clip in clips:
            clip.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating video: {e}")
        return False


def add_transitions(image_paths, output_path, transition_duration=0.5, fps=24):
    """
    Create video with crossfade transitions between images
    
    Args:
        image_paths: List of image file paths
        output_path: Output video file path
        transition_duration: Crossfade duration in seconds
        fps: Frames per second
        
    Returns:
        Boolean indicating success
    """
    try:
        from moviepy.editor import ImageClip, CompositeVideoClip
    except ImportError:
        print("MoviePy not available for transitions")
        return False
    
    try:
        clips = []
        shot_duration = 3.0
        current_time = 0
        
        for i, img_path in enumerate(image_paths):
            if not Path(img_path).exists():
                continue
            
            clip = ImageClip(img_path).set_duration(shot_duration)
            
            # Add crossfade for clips after the first
            if i > 0:
                clip = clip.crossfadein(transition_duration)
                clip = clip.set_start(current_time - transition_duration)
                current_time += (shot_duration - transition_duration)
            else:
                clip = clip.set_start(current_time)
                current_time += shot_duration
            
            clips.append(clip)
        
        # Composite all clips
        final_video = CompositeVideoClip(clips)
        
        # Write output
        final_video.write_videofile(
            output_path,
            fps=fps,
            codec='libx264',
            verbose=False,
            logger=None
        )
        
        # Cleanup
        final_video.close()
        for clip in clips:
            clip.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating transitions: {e}")
        return False