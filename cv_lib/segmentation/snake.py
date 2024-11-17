# cv_lib/segmentation/snake.py
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline  # Changed from interp2d
from ..core.image import Image
import cv2

def _is_in_jupyter() -> bool:
    """Check if code is running in Jupyter notebook/lab."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False
    except (ImportError, AttributeError, NameError):
        return False

@dataclass
class SnakeParams:
    """Parameters for snake evolution."""
    alpha: float = 0.1  # Tension parameter
    beta: float = 0.1   # Rigidity parameter
    gamma: float = .05  # Step size
    max_iters: int = 100
    convergence_threshold: float = 0.1

class Snake:
    """Active contour model (snake) implementation for image segmentation."""
    
    def __init__(self, image: Image, params: Optional[SnakeParams] = None):
        """Initialize snake with image and parameters."""
        if not (image.is_grayscale() or image.is_color()):
            raise ValueError("Image must be grayscale or color")
        
        self.original_image = image
        if image.is_color():
            print("Converting to grayscale for processing...")
            self.image = image.to_grayscale()
        else:
            self.image = image
            
        self.params = params or SnakeParams()
        self.snake_points: Optional[np.ndarray] = None
        self.evolution_history: List[np.ndarray] = []
        self.in_jupyter = _is_in_jupyter()
    
    def _visualize_forces(self, force_field: np.ndarray, edge_map: np.ndarray):
        """Debug visualization of force field and edge map."""
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(self.image.data[..., 0], cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')
        
        # Edge map
        plt.subplot(132)
        plt.imshow(edge_map, cmap='jet')
        plt.title('Edge Map')
        plt.axis('off')
        
        # Force field
        plt.subplot(133)
        step = 10  # Sample every 10 pixels for visualization
        Y, X = np.mgrid[0:force_field.shape[0]:step, 0:force_field.shape[1]:step]
        U = force_field[::step, ::step, 0]
        V = force_field[::step, ::step, 1]
        plt.quiver(X, Y, U, V, scale=50)
        plt.imshow(self.image.data[..., 0], cmap='gray', alpha=0.3)
        plt.title('Force Field')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _compute_external_force(self) -> np.ndarray:
        """Compute external force field based on image gradient."""
        print(f"Computing forces. Image shape: {self.image.data.shape}, dtype: {self.image.data.dtype}")
        print(f"Image range: [{self.image.data.min()}, {self.image.data.max()}]")
        
        # Ensure we're working with grayscale
        img_data = self.image.data[..., 0]
        
        # Compute gradient with Sobel
        dx = cv2.Sobel(img_data, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(img_data, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        print(f"Gradient magnitude range: [{gradient_magnitude.min()}, {gradient_magnitude.max()}]")
        
        # Gaussian blur to smooth force field
        from ..core.ops import Convolution, KernelType
        conv = Convolution()
        
        # First smoothing of gradient magnitude
        gradient_magnitude = conv.apply_kernel(
            Image(gradient_magnitude[..., np.newaxis], normalize=False),
            KernelType.GAUSSIAN,
            size=5,
            sigma=2.0
        ).data[..., 0]
        
        # Normalize gradient magnitude
        gradient_magnitude = gradient_magnitude / (gradient_magnitude.max() + 1e-8)
        
        # Create edge map
        edge_map = gradient_magnitude
        
        # Double smoothing for better force field
        edge_map = conv.apply_kernel(
            Image(edge_map[..., np.newaxis], normalize=False),
            KernelType.GAUSSIAN,
            size=15,
            sigma=3.0
        ).data[..., 0]
        
        # Compute forces as negative gradient of edge map
        fx = cv2.Sobel(edge_map, cv2.CV_64F, 1, 0, ksize=3)
        fy = cv2.Sobel(edge_map, cv2.CV_64F, 0, 1, ksize=3)
        
        # Normalize forces
        magnitude = np.sqrt(fx**2 + fy**2)
        max_magnitude = magnitude.max()
        if max_magnitude > 0:
            fx = fx / max_magnitude
            fy = fy / max_magnitude
        
        force_field = np.stack([-fx, -fy], axis=-1)
        
        # Visualize the forces and edge map
        self._visualize_forces(force_field, edge_map)
        
        return force_field
    
    def _get_matrix_A(self, n_points: int) -> np.ndarray:
        """Construct matrix A for internal forces."""
        alpha, beta = self.params.alpha, self.params.beta
        A = np.zeros((n_points, n_points))

        # Indices for wrapping around
        idx = np.arange(n_points)
        A[idx, idx] = 2 * alpha + 6 * beta
        A[idx, (idx - 1) % n_points] = -alpha - 4 * beta
        A[idx, (idx - 2) % n_points] = beta
        A[idx, (idx + 1) % n_points] = -alpha - 4 * beta
        A[idx, (idx + 2) % n_points] = beta

        return A
    
    def initialize_interactive(self):
        """Initialize interactive snake drawing."""
        print('test 2')
        if not self.in_jupyter:
            plt.ion()
        
        plt.clf()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Show grayscale image for snake evolution
        if self.original_image.is_color():
            # Show original color image but use grayscale for processing
            self.ax.imshow(self.original_image.data)
        else:
            self.ax.imshow(self.image.data[..., 0], cmap='gray')
            
        self.ax.set_title("Click to draw contour. Press Enter when done.")
        
        self.points: List[Tuple[float, float]] = []
        # Create two line artists: one for the current points and one for preview
        self.line, = self.ax.plot([], [], 'y-', linewidth=2)
        self.preview_line, = self.ax.plot([], [], 'y--', linewidth=1, alpha=0.5)
        
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        
        if self.in_jupyter:
            plt.show()
        else:
            plt.show(block=True)
    
    def _update_preview(self, x, y):
        """Update the preview line connecting current point to mouse position."""
        if len(self.points) > 0:
            preview_x = [self.points[-1][0], x]
            preview_y = [self.points[-1][1], y]
            self.preview_line.set_data(preview_x, preview_y)
            
            # Also show preview of closing the contour if we have more than 2 points
            if len(self.points) > 2:
                preview_x = [x, self.points[0][0]]
                preview_y = [y, self.points[0][1]]

    def _on_mouse_move(self, event):
        """Handle mouse movement for preview line."""
        if event.inaxes != self.ax or len(self.points) == 0:
            return
            
        self._update_preview(event.xdata, event.ydata)
        if self.in_jupyter:
            self.fig.canvas.draw()
        else:
            self.fig.canvas.draw_idle()
    
    def _on_click(self, event):
        """Handle mouse clicks for drawing."""
        if event.inaxes != self.ax:
            return
        
        self.points.append((event.xdata, event.ydata))
        x = [p[0] for p in self.points]
        y = [p[1] for p in self.points]
        
        # If we have more than one point, show the line
        if len(self.points) > 1:
            self.line.set_data(x, y)
            
            # If we have more than two points, show preview of closed contour
            if len(self.points) > 2:
                x_closed = x + [x[0]]
                y_closed = y + [y[0]]
                self.line.set_data(x_closed, y_closed)
        
        if self.in_jupyter:
            self.fig.canvas.draw()
        else:
            self.fig.canvas.draw_idle()
    
    def _on_key(self, event):
        """Handle key presses."""
        if event.key == 'enter':
            # Close the contour if we have at least 3 points
            if len(self.points) > 2:
                # Add first point to close the contour if not already closed
                if self.points[-1] != self.points[0]:
                    self.points.append(self.points[0])
                    
                x = [p[0] for p in self.points]
                y = [p[1] for p in self.points]
                self.line.set_data(x, y)
                
                if self.in_jupyter:
                    self.fig.canvas.draw()
                else:
                    self.fig.canvas.draw_idle()
                
                # Convert to numpy array
                self.snake_points = np.array(self.points)
                plt.close(self.fig)
                
                if not self.in_jupyter:
                    plt.ioff()
            else:
                print("Please add at least 3 points to create a contour.")
    
    def evolve(self, animate: bool = True) -> np.ndarray:
        """Evolve the snake to minimize energy."""
        if self.snake_points is None:
            raise ValueError("Snake not initialized. Call initialize_interactive first.")
            
        # Debug: Show current image
        plt.figure(figsize=(10, 10))
        plt.imshow(self.original_image.data)
        plt.plot(self.snake_points[:, 0], self.snake_points[:, 1], 'r-', label='Initial contour')
        plt.title('Initial Snake Position')
        plt.legend()
        plt.show()
            
        # Compute external force field
        print("Computing external forces...")
        force_field = self._compute_external_force()
        
        # Get matrix A for internal forces
        n_points = len(self.snake_points)
        A = self._get_matrix_A(n_points)
        
        # Initialize solution matrix
        B = np.eye(n_points) + self.params.gamma * A
        B_inv = np.linalg.inv(B)
        
        # Evolution loop
        self.evolution_history = [self.snake_points.copy()]
        
        # Setup interpolation
        x = np.arange(force_field.shape[1])
        y = np.arange(force_field.shape[0])
        interpolator_x = RectBivariateSpline(y, x, force_field[..., 0])
        interpolator_y = RectBivariateSpline(y, x, force_field[..., 1])
        
        if animate:
            if not self.in_jupyter:
                plt.ion()
            
            fig, ax = plt.subplots(figsize=(10, 10))
            if self.original_image.is_color():
                ax.imshow(self.original_image.data)
            else:
                ax.imshow(self.image.data[..., 0], cmap='gray')
                
            # Plot initial contour
            current_points = np.vstack((self.snake_points, self.snake_points[0:1]))
            line, = ax.plot(current_points[:, 0], current_points[:, 1], 'y-', linewidth=2)
            plt.show(block=False)
        
        # Add momentum terms
        prev_dx = np.zeros(n_points)
        prev_dy = np.zeros(n_points)
        momentum = 0.3  # Momentum coefficient
        
        for _ in range(self.params.max_iters):
            prev_points = self.snake_points.copy()
            
            # Get external forces at snake points
            fx = interpolator_x.ev(self.snake_points[:, 1], self.snake_points[:, 0])
            fy = interpolator_y.ev(self.snake_points[:, 1], self.snake_points[:, 0])
            
            # Add momentum
            fx = fx + momentum * prev_dx
            fy = fy + momentum * prev_dy
            
            # Update snake points
            new_x = B_inv @ (self.snake_points[:, 0] + self.params.gamma * fx)
            new_y = B_inv @ (self.snake_points[:, 1] + self.params.gamma * fy)
            
            # Store momentum terms
            prev_dx = new_x - self.snake_points[:, 0]
            prev_dy = new_y - self.snake_points[:, 1]
            
            # Update positions
            self.snake_points = np.column_stack([new_x, new_y])
            
            # Ensure snake points stay within image bounds
            self.snake_points[:, 0] = np.clip(self.snake_points[:, 0], 0, self.image.shape[1] - 1)
            self.snake_points[:, 1] = np.clip(self.snake_points[:, 1], 0, self.image.shape[0] - 1)
            
            # Store for history
            self.evolution_history.append(self.snake_points.copy())
            
            # Update animation
            if animate:
                current_points = np.vstack((self.snake_points, self.snake_points[0:1]))
                line.set_data(current_points[:, 0], current_points[:, 1])
                if self.in_jupyter:
                    fig.canvas.draw()
                else:
                    fig.canvas.draw_idle()
                plt.pause(0.05)
            
            # Check convergence
            if np.mean(np.abs(self.snake_points - prev_points)) < self.params.convergence_threshold:
                break
        
        if animate:
            if not self.in_jupyter:
                plt.ioff()
            plt.close(fig)
        
        return self.snake_points
    
    def get_mask(self) -> np.ndarray:
        """Get binary mask of region inside snake."""
        from matplotlib.path import Path
        
        h, w = self.image.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        
        path = Path(self.snake_points)
        grid = path.contains_points(points)
        mask = grid.reshape(h, w)
        
        return mask
    
    def save_result(self, filepath: str, mask_only: bool = False):
        """Save segmentation result."""
        if mask_only:
            mask = self.get_mask()
            # Convert boolean mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            Image(mask_uint8[..., np.newaxis], normalize=False).save(filepath)
        else:
            # Draw snake on original image
            result = self.original_image.copy()
            h, w = self.image.shape[:2]
            points = np.round(self.snake_points).astype(int)
            points = np.clip(points, 0, [w-1, h-1])
            
            # Draw yellow contour
            from skimage.draw import polygon_perimeter
            rr, cc = polygon_perimeter(points[:, 1], points[:, 0], shape=result.shape[:2])
            if result.is_color():
                result.data[rr, cc] = [1, 1, 0]  # Yellow
            else:
                result.data[rr, cc] = 1
            
            result.save(filepath)