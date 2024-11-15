# cv_lib/filters/noise.py
import numpy as np
from enum import Enum
from typing import Optional
from ..core.image import Image

class NoiseGenerator:
    """Class for adding noise to images."""
    
    @staticmethod
    def add_gaussian_noise(image: Image, var: float = 0.01) -> Image:
        """Add Gaussian noise to image."""
        print(f"\nAdding noise to image with shape: {image.data.shape}")
        noise = np.random.normal(0, np.sqrt(var), image.data.shape)
        noisy_data = np.clip(image.data + noise, 0, 1).astype(np.float32)
        
        # Ensure proper shape
        if len(noisy_data.shape) == 2:
            noisy_data = noisy_data[..., np.newaxis]
        print(f"Noisy data shape: {noisy_data.shape}")
        
        noisy_img = Image(noisy_data, normalize=False)
        print(f"Created noisy image: color={noisy_img.is_color()}, gray={noisy_img.is_grayscale()}")
        return noisy_img