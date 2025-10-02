"""
Diffusion model wrapper for image generation
"""

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image


class DiffusionGenerator:
    """Wrapper for Stable Diffusion image generation"""
    
    def __init__(self, config):
        """
        Initialize diffusion model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = self._get_device()
        self.pipeline = None
        
        print(f"Using device: {self.device}")
        self._initialize_pipeline()
    
    def _get_device(self):
        """Determine available compute device"""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _initialize_pipeline(self):
        """Load and configure the diffusion pipeline"""
        print("Loading diffusion model")
        
        # Set appropriate dtype
        if self.device == 'cpu':
            dtype = torch.float32
            print("CPU detected, using float32 precision")
        else:
            dtype = torch.float16
            print(f"{self.device.upper()} detected, using float16 precision")
        
        # Load pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config['model_name'],
            torch_dtype=dtype,
            safety_checker=None,
            use_safetensors=True
        )
        
        # Configure scheduler
        self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        # Apply optimizations
        self._apply_optimizations()
        
        print("Model initialization complete")
    
    def _apply_optimizations(self):
        """Apply device-specific performance optimizations"""
        if self.device == 'cuda':
            # Enable attention slicing for memory efficiency
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
                print("Attention slicing enabled")
            
            # Try to enable xformers
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("XFormers optimization enabled")
            except ImportError:
                print("XFormers not available, install with: pip install xformers")
            except Exception as e:
                print(f"Could not enable XFormers: {e}")
    
    def generate_image(self, prompt, output_file):
        """
        Generate single image from text prompt
        
        Args:
            prompt: Text description
            output_file: Path to save generated image
            
        Returns:
            Path to saved image
        """
        # Enhance prompt with quality modifiers
        enhanced_prompt = self._enhance_prompt(prompt)
        negative_prompt = self._get_negative_prompt()
        
        # Setup generator for reproducibility
        generator = None
        if self.config.get('random_seed') is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.config['random_seed'])
        
        # Generate image
        with torch.no_grad():
            result = self.pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                width=self.config['resolution']['width'],
                height=self.config['resolution']['height'],
                num_inference_steps=self.config['inference_steps'],
                guidance_scale=self.config['guidance_scale'],
                generator=generator
            )
        
        # Save image
        image = result.images[0]
        image.save(output_file)
        
        return output_file
    
    def _enhance_prompt(self, base_prompt):
        """
        Add quality enhancement terms to prompt
        
        Args:
            base_prompt: Original prompt text
            
        Returns:
            Enhanced prompt string
        """
        quality_modifiers = [
            "cinematic composition",
            "film photography",
            "dramatic lighting",
            "professional color grading",
            "shallow depth of field"
        ]
        
        enhanced = f"{base_prompt}, {', '.join(quality_modifiers)}"
        return enhanced
    
    def _get_negative_prompt(self):
        """Return negative prompt for quality control"""
        return "blurry, low quality, distorted, amateur, ugly, deformed, text, watermark, signature"
    
    def cleanup(self):
        """Release model resources and free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Model resources released")