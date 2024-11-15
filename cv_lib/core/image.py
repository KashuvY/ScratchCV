# cv_lib/core/image.py
import numpy as np
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
    
    # ... rest of the methods remain the same ...
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
    
    @NotImplementedError
    def resize(self, new_height: int, new_width: int) -> 'Image':
        """This method is planned but not implemented yet."""
        raise NotImplementedError("This feature is coming soon")
    
    @NotImplementedError
    def rotate(self, angle: float) -> 'Image':
        """This method is planned but not implemented yet."""
        raise NotImplementedError("This feature is coming soon")
    
    @NotImplementedError
    def flip(self, horizontal: bool = True) -> 'Image':
        """This method is planned but not implemented yet."""
        raise NotImplementedError("This feature is coming soon")
    
    @NotImplementedError
    def crop(self, x: int, y: int, width: int, height: int) -> 'Image':
        """This method is planned but not implemented yet."""
        raise NotImplementedError("This feature is coming soon")