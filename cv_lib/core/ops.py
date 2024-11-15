import numpy as np
from typing import Union, Tuple, Optional
from enum import Enum
from .image import Image

class KernelType(Enum):
    """Enumeration of supported kernel types."""
    GAUSSIAN = "gaussian"
    SOBEL = "sobel"
    LAPLACIAN = "laplacian"
    SHARPEN = "sharpen"
    BOX_BLUR = "box_blur"

class Kernel:
    """Class for generating different types of kernels."""
    
    @staticmethod
    def gaussian(size: int = 3, sigma: float = 1.0) -> np.ndarray:
        """Generate a Gaussian kernel."""
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        return kernel / np.sum(kernel)
    
    @staticmethod
    def sobel(direction: str = 'x') -> np.ndarray:
        """Generate Sobel edge detection kernel."""
        if direction.lower() == 'x':
            return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        elif direction.lower() == 'y':
            return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        raise ValueError("Direction must be 'x' or 'y'")
    
    @staticmethod
    def laplacian() -> np.ndarray:
        """Generate Laplacian kernel for edge detection."""
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    
    @staticmethod
    def sharpen() -> np.ndarray:
        """Generate sharpening kernel."""
        return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    
    @staticmethod
    def box_blur(size: int = 3) -> np.ndarray:
        """Generate box blur kernel."""
        return np.ones((size, size)) / (size * size)

class Convolution:
    """Implements different types of convolutions for image processing."""
    
    def __init__(self):
        self.kernel_generators = {
            KernelType.GAUSSIAN: Kernel.gaussian,
            KernelType.SOBEL: Kernel.sobel,
            KernelType.LAPLACIAN: Kernel.laplacian,
            KernelType.SHARPEN: Kernel.sharpen,
            KernelType.BOX_BLUR: Kernel.box_blur
        }
    
    def convolve2d(self, image: 'Image', kernel: np.ndarray, 
                   padding: str = 'same') -> 'Image':
        """Apply 2D convolution to an image."""
        if image.is_color():
            # Process each channel separately
            result = np.dstack([
                self._convolve2d_single_channel(image.data[..., i], kernel, padding)
                for i in range(image.shape[2])
            ])
        else:
            result = self._convolve2d_single_channel(
                image.data[..., 0], kernel, padding
            )[..., np.newaxis]
        
        return Image(result, normalize=False)
    
    def _convolve2d_single_channel(self, channel: np.ndarray, kernel: np.ndarray,
                                  padding: str = 'same') -> np.ndarray:
        """Internal method for 2D convolution on a single channel."""
        if kernel.ndim != 2:
            raise ValueError("Kernel must be 2D")
            
        kernel = np.flipud(np.fliplr(kernel))
        k_rows, k_cols = kernel.shape
        
        if padding == 'same':
            pad_rows = (k_rows - 1) // 2
            pad_cols = (k_cols - 1) // 2
            padded = np.pad(channel, ((pad_rows, pad_rows), (pad_cols, pad_cols)),
                           mode='reflect')
        else:  # 'valid'
            padded = channel
            
        out_rows = padded.shape[0] - k_rows + 1
        out_cols = padded.shape[1] - k_cols + 1
        
        output = np.zeros((out_rows, out_cols), dtype=np.float32)
        for i in range(out_rows):
            for j in range(out_cols):
                output[i, j] = np.sum(
                    padded[i:i+k_rows, j:j+k_cols] * kernel
                )
                
        return output
    
    def apply_kernel(self, image: 'Image', kernel_type: KernelType,
                    **kwargs) -> 'Image':
        """Apply a specific type of kernel to an image."""
        if kernel_type not in self.kernel_generators:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
            
        kernel = self.kernel_generators[kernel_type](**kwargs)
        return self.convolve2d(image, kernel)
    
    def sharpen(self, image: 'Image', amount: float = 1.0) -> 'Image':
        """Sharpen an image using the unsharp mask technique."""
        # Create a blurred version
        blurred = self.apply_kernel(image, KernelType.GAUSSIAN, size=3, sigma=1.0)
        
        # Calculate the difference
        detail = Image(image.data - blurred.data, normalize=False)
        
        # Add scaled detail back to original
        sharpened = Image(image.data + amount * detail.data, normalize=False)
        
        return sharpened.clip()
    
    def deblur(self, image: 'Image', strength: float = 1.0) -> 'Image':
        """Deblur an image using the Laplacian operator."""
        # Apply Laplacian to detect edges
        laplacian = self.apply_kernel(image, KernelType.LAPLACIAN)
        
        # Subtract scaled Laplacian from original
        deblurred = Image(image.data - strength * laplacian.data, normalize=False)
        
        return deblurred.clip()