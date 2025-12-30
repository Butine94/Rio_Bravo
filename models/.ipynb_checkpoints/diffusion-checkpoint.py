import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler
)
from PIL import Image
import os
import cv2
import numpy as np
from typing import List, Dict, Optional

class CinematicDiffusionModel:
    """High-quality image generation for cinematic shots with ControlNet and LoRA"""
    
    def __init__(self, config: Dict):
        self.config = config['diffusion']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.controlnet_enabled = self.config['use_controlnet']
        self.lora_enabled = self.config['use_lora']
        self._load_model()
    
    def _load_model(self):
        """Load and optimize the diffusion model"""
        print(f"Loading model on {self.device}...")
        
        dtype = torch.float32 if self.device == "cpu" else (
            torch.float16 if self.config['dtype'] == 'fp16' else torch.float32
        )
        
        if self.controlnet_enabled:
            print("Loading ControlNet...")
            controlnet = ControlNetModel.from_pretrained(
                self.config['controlnet_model'],
                torch_dtype=dtype
            )
            
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.config['base_model'],
                controlnet=controlnet,
                torch_dtype=dtype,
                use_safetensors=True
            )
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.config['base_model'],
                torch_dtype=dtype,
                use_safetensors=True
            )
        
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to(self.device)
        
        if self.lora_enabled:
            self._load_lora()
        
        if self.device == "cuda":
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
            
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("XFormers enabled")
            except:
                pass
        
        print("Model loaded")
    
    def _load_lora(self):
        """Load LoRA weights"""
        lora_path = self.config['lora_path']
        
        if not os.path.exists(lora_path):
            print(f"Warning: LoRA not found: {lora_path}")
            return
        
        try:
            self.pipe.load_lora_weights(lora_path)
            print(f"LoRA loaded: {lora_path}")
        except Exception as e:
            print(f"LoRA load failed: {e}")
    
    def _create_depth_map(self, image_path: Optional[str] = None, 
                          width: int = 512, height: int = 512) -> Image.Image:
        """Create depth map for ControlNet"""
        if image_path and os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (width, height))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.linspace(255, 128, height, dtype=np.uint8)
            gray = np.tile(gray.reshape(-1, 1), (1, width))
        
        depth_map = cv2.GaussianBlur(gray, (5, 5), 0)
        return Image.fromarray(depth_map).convert("RGB")
    
    def generate_shots(self, shots: List[Dict], output_dir: str, 
                       use_depth_consistency: bool = False) -> List[Dict]:
        """Generate high-quality images for all shots"""
        if not self.pipe:
            raise RuntimeError("Model not loaded")
        
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config['seed'])
        
        updated_shots = []
        previous_depth = None
        
        for i, shot in enumerate(shots):
            print(f"Generating shot {i+1}/{len(shots)}: {shot['shot_type']}")
            
            prompt = self._enhance_prompt(shot['prompt'])
            
            gen_kwargs = {
                'prompt': prompt,
                'negative_prompt': "blurry, low quality, distorted, amateur, ugly, deformed, cartoon, anime",
                'height': self.config['height'],
                'width': self.config['width'],
                'num_inference_steps': self.config['num_inference_steps'],
                'guidance_scale': self.config['guidance_scale'],
                'generator': generator
            }
            
            if self.lora_enabled:
                gen_kwargs['cross_attention_kwargs'] = {"scale": self.config['lora_scale']}
            
            if self.controlnet_enabled:
                depth_image = previous_depth if (use_depth_consistency and previous_depth) else (
                    self._create_depth_map(width=self.config['width'], height=self.config['height'])
                )
                gen_kwargs['image'] = depth_image
                gen_kwargs['controlnet_conditioning_scale'] = self.config['controlnet_scale']
            
            with torch.no_grad():
                result = self.pipe(**gen_kwargs)
            
            image = result.images[0]
            image_path = os.path.join(output_dir, f"shot_{i+1}.png")
            image.save(image_path)
            
            if use_depth_consistency and self.controlnet_enabled:
                previous_depth = self._create_depth_map(image_path)
            
            shot['image_path'] = image_path
            updated_shots.append(shot)
            
            print(f"Saved: {image_path}")
        
        return updated_shots
    
    def _enhance_prompt(self, base_prompt: str) -> str:
        """Enhance prompt for cinematic quality"""
        enhancements = [
            "highly detailed",
            "8k resolution",
            "professional photography",
            "perfect composition",
            "atmospheric lighting",
            "cinematic color grading"
        ]
        
        if self.lora_enabled and self.config['style_tokens']:
            enhancements.extend(self.config['style_tokens'])
        
        return f"{base_prompt}, {', '.join(enhancements)}"
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.pipe:
            del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()