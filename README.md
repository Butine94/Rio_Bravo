Rio Bravo
AI film generator that converts screenplay text into cinematic storyboards using Stable Diffusion with ControlNet composition control and LoRA style consistency.

Features:
ControlNet Integration - Spatial composition control for consistent framing and depth
LoRA Style Application - Cinematic aesthetics (film grain, lighting, color grading)
Script Parsing - Extracts visual elements from screenplay text
Professional Quality - High-resolution outputs with optimized inference
Video Compilation - Optional MP4 generation from image sequences


# Install dependencies
pip install -r requirements.txt

# Generate storyboards from script
python generate.py

# With video output
python generate.py --video

# Custom settings
python generate.py --script data/input.txt --num-shots 10 --depth-consistency


Example Output
[Add 2-3 generated storyboard images here]
Generated from the Rio Bravo script: Mojave Desert trains sequence with cinematic composition and film aesthetic.

Rio_Bravo/
├── generate.py              # Main pipeline
├── diffusion_model.py       # Core model with ControlNet/LoRA
├── config.yaml              # Configuration settings
├── data/
│   └── input.txt           # Script input
├── utils/
│   ├── text_processing.py  # Script parsing
│   ├── io_utils.py         # File operations
│   └── controlnet_utils.py # Preprocessing
└── outputs/                 # Generated images