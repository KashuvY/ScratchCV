import numpy as np

'''
class Kernel:
    """Base class for different types of kernels"""
    GAUSSIAN = 'gaussian'
    SOBEL = 'sobel'
    LAPLACIAN = 'laplacian'

class Convolution:
    """Implements different types of convolutions"""
    def conv2d(self, image: Image, kernel: np.ndarray) -> Image
    def separable_conv2d(self, image: Image, kernel_x: np.ndarray, kernel_y: np.ndarray) -> Image
'''

class Convolution:
    """Base class for convolution operations."""
    
    @staticmethod
    def convolve2d(image: np.ndarray, kernel: np.ndarray, 
                   padding: str = 'same') -> np.ndarray:
        """2D convolution implementation from scratch.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        kernel : np.ndarray
            Convolution kernel
        padding : str, optional
            Padding type ('same' or 'valid'), by default 'same'
            
        Returns
        -------
        np.ndarray
            Convolved image
        """
        if kernel.ndim != 2:
            raise ValueError("Kernel must be 2D")
            
        kernel = np.flipud(np.fliplr(kernel))  # Flip kernel for convolution
        k_rows, k_cols = kernel.shape
        
        # Handle padding
        if padding == 'same':
            pad_rows = (k_rows - 1) // 2
            pad_cols = (k_cols - 1) // 2
            padded = np.pad(image, ((pad_rows, pad_rows), (pad_cols, pad_cols)),
                          mode='reflect')
        else:  # 'valid'
            padded = image
            
        # Get output dimensions
        out_rows = padded.shape[0] - k_rows + 1
        out_cols = padded.shape[1] - k_cols + 1
        
        # Perform convolution
        output = np.zeros((out_rows, out_cols), dtype=np.float32)
        for i in range(out_rows):
            for j in range(out_cols):
                output[i, j] = np.sum(
                    padded[i:i+k_rows, j:j+k_cols] * kernel
                )
                
        return output
