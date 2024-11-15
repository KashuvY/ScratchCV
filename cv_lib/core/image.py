# cv_lib/core/image.py
import numpy as np
import cv2
from typing import Union, Tuple, NoReturn
import matplotlib.pyplot as plt
from skimage import io, img_as_float32

class Image:
    """Core image class for the CV library."""
    # TODO: Add support for resizing images
    # TODO: Add support for rotating imgaes
    # TODO: Add support for flipping images
    # TODO: Add support for cropping images
    
    def __init__(self, data: Union[str, np.ndarray], normalize: bool = True):
        """Initialize an image from numpy array or file path."""
        if isinstance(data, str):
            print(f"Loading image from {data}")
            try:
                self._data = io.imread(data)
                print(f"Initial load shape: {self._data.shape}, dtype: {self._data.dtype}")
            except Exception as e:
                print(f"Error loading image: {e}")
                raise
        elif isinstance(data, np.ndarray):
            self._data = data.copy()
            print(f"Array input shape: {self._data.shape}, dtype: {self._data.dtype}")
        else:
            raise ValueError("Data must be filepath or numpy array")
        
        # Convert to float32 and normalize if needed
        if normalize:
            self._data = img_as_float32(self._data)
        
        # Ensure proper shape for grayscale images
        if len(self._data.shape) == 2:
            print("Converting 2D array to 3D")
            self._data = self._data[..., np.newaxis]
            
        print(f"Final data shape: {self._data.shape}, dtype: {self._data.dtype}")
    
    @property
    def data(self) -> np.ndarray:
        """Get image data."""
        return self._data
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get image shape."""
        return self._data.shape
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return self._data.ndim
    
    @property
    def dtype(self):
        """Get data type."""
        return self._data.dtype
    
    def is_color(self) -> bool:
        """Check if image is color (3D array with 3 or 4 channels)."""
        result = len(self._data.shape) == 3 and self._data.shape[2] in [3, 4]
        print(f"is_color check: shape={self._data.shape}, result={result}")
        return result
    
    def is_grayscale(self) -> bool:
        """Check if image is grayscale (2D array or 3D with 1 channel)."""
        result = len(self._data.shape) == 2 or (
            len(self._data.shape) == 3 and self._data.shape[2] == 1
        )
        print(f"is_grayscale check: shape={self._data.shape}, result={result}")
        return result
    
    def to_grayscale(self) -> 'Image':
        """Convert to grayscale if color."""
        if self.is_grayscale():
            return self
        # Using standard RGB to grayscale conversion weights
        weights = np.array([0.299, 0.587, 0.114])
        gray_data = np.dot(self._data[..., :3], weights)
        return Image(gray_data[..., np.newaxis], normalize=False)
    
    def clip(self) -> 'Image':
        """Clip values to valid range [0,1]."""
        return Image(np.clip(self._data, 0, 1), normalize=False)
    
    def save(self, filepath: str) -> None:
        """Save image to file."""
        save_data = self._data
        if save_data.dtype == np.float32:
            save_data = (save_data * 255).clip(0, 255).astype(np.uint8)
        if save_data.shape[-1] == 1:  # Grayscale
            save_data = save_data.squeeze()
        # Convert RGB to BGR for OpenCV
        if len(save_data.shape) == 3 and save_data.shape[2] == 3:
            save_data = cv2.cvtColor(save_data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, save_data)
    
    def show(self) -> None:
        """Display the image."""
        plt.figure(figsize=(8, 8))
        plt.imshow(self._data)
        plt.axis('off')
        plt.show()
    
    def copy(self) -> 'Image':
        """Create a deep copy of the image."""
        return Image(self._data.copy(), normalize=False)

    def __array__(self) -> np.ndarray:
        """Numpy array interface."""
        return self._data
    
    def resize(self, new_height: int, new_width: int) -> 'Image':
        """Resize the image using improved interpolation.
        
        Args:
            new_height (int): Target height in pixels
            new_width (int): Target width in pixels
            
        Returns:
            Image: Resized image with improved quality
        """
        if new_height <= 0 or new_width <= 0:
            raise ValueError("New dimensions must be positive")
            
        old_height, old_width = self._data.shape[:2]
        num_channels = self._data.shape[2]
        
        # Compute scale factors
        x_ratio = old_width / new_width
        y_ratio = old_height / new_height
        
        # Create coordinate matrices for the output image with subpixel precision
        x_coords = np.linspace(0, old_width - 1, new_width)
        y_coords = np.linspace(0, old_height - 1, new_height)
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
        
        # Get the four nearest neighbors for each point
        x0 = np.floor(x_mesh).astype(int)
        x1 = np.minimum(x0 + 1, old_width - 1)
        y0 = np.floor(y_mesh).astype(int)
        y1 = np.minimum(y0 + 1, old_height - 1)
        
        # Calculate interpolation weights with improved precision
        wx = x_mesh - x0
        wy = y_mesh - y0
        
        # Add small epsilon to prevent floating point errors
        wx = wx.clip(0, 1)
        wy = wy.clip(0, 1)
        
        # Create output array with proper dtype
        resized = np.zeros((new_height, new_width, num_channels), dtype=self._data.dtype)
        
        def cubic_weight(x):
            """Cubic interpolation weight function using numpy.where()."""
            x = np.abs(x)
            result = np.zeros_like(x)
            
            # Region 1: 0 <= x <= 1
            mask1 = x <= 1
            result = np.where(mask1, 1.5 * x**3 - 2.5 * x**2 + 1, result)
            
            # Region 2: 1 < x < 2
            mask2 = (x > 1) & (x < 2)
            result = np.where(mask2, -0.5 * x**3 + 2.5 * x**2 - 4 * x + 2, result)
            
            return result
            
        # Perform improved interpolation for each channel
        for c in range(num_channels):
            # Get values of four neighbors
            v00 = self._data[y0, x0, c]
            v01 = self._data[y0, x1, c]
            v10 = self._data[y1, x0, c]
            v11 = self._data[y1, x1, c]
            
            # Apply bicubic-inspired weighting for smoother interpolation
            wx_cubic = wx  # Use linear interpolation weights instead
            wy_cubic = wy
            
            # Compute weighted averages
            top = v00 * (1 - wx_cubic) + v01 * wx_cubic
            bottom = v10 * (1 - wx_cubic) + v11 * wx_cubic
            resized[..., c] = top * (1 - wy_cubic) + bottom * wy_cubic
        
        # Apply a subtle sharpening filter
        if min(x_ratio, y_ratio) > 1:  # Only when downscaling
            kernel = np.array([[-0.1, -0.1, -0.1],
                             [-0.1,  1.8, -0.1],
                             [-0.1, -0.1, -0.1]])
            
            padded = np.pad(resized, ((1, 1), (1, 1), (0, 0)), mode='edge')
            filtered = np.zeros_like(resized)
            
            for i in range(resized.shape[0]):
                for j in range(resized.shape[1]):
                    for c in range(num_channels):
                        window = padded[i:i+3, j:j+3, c]
                        filtered[i, j, c] = np.sum(window * kernel)
            
            resized = np.clip(filtered, 0, 1)
        
        return Image(resized, normalize=False)
    
    def rotate(self, angle: float) -> 'Image':
        """This method is planned but not implemented yet."""
        raise NotImplementedError("This feature is coming soon")
    
    def flip(self, horizontal: bool = True) -> 'Image':
        """This method is planned but not implemented yet."""
        raise NotImplementedError("This feature is coming soon")
    
    def crop(self, x: int, y: int, width: int, height: int) -> 'Image':
        """This method is planned but not implemented yet."""
        raise NotImplementedError("This feature is coming soon")