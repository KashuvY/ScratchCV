# CV Lib

A computer vision library built from scratch for learning purposes.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ScratchCV.git
cd ScratchCV

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in editable mode
pip install -e .
```

## Usage

```python
from cv_lib import Image, NoiseGenerator, TVDenoiser, BoundaryCondition

# Load and process an image
img = Image("path/to/image.jpg")
noisy_img = NoiseGenerator.add_gaussian_noise(img, var=0.01)
denoiser = TVDenoiser(mu=0.1, lambda_=0.1)
denoised_img = denoiser.denoise(noisy_img)
```

## Development

This project uses pip in editable mode for development. Any changes to the source code will be immediately reflected without reinstalling.