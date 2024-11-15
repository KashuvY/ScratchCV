# cv_lib/filters/denoising.py
import numpy as np
from enum import Enum
from typing import Tuple
from ..core.image import Image

class BoundaryCondition(Enum):
    """Supported boundary conditions."""
    DIRICHLET_ZERO = 'dirichlet_zero'
    NEUMANN = 'neumann'
    PERIODIC = 'periodic'
    DIRICHLET_COPY = 'dirichlet_copy'

class TVDenoiser:
    """Total Variation denoising using Split Bregman method."""
    
    def __init__(self, mu: float = 0.1, lambda_: float = 0.1, 
                 gamma: float = 1.0, max_iters: int = 100,
                 boundary_condition: BoundaryCondition = BoundaryCondition.NEUMANN):
        """Initialize denoiser with parameters."""
        self.mu = mu
        self.lambda_ = lambda_
        self.gamma = gamma
        self.max_iters = max_iters
        self.boundary_condition = boundary_condition
    
    def _pad_with_boundary_conditions(self, f: np.ndarray) -> np.ndarray:
        """Pad array according to specified boundary conditions."""
        if self.boundary_condition == BoundaryCondition.DIRICHLET_ZERO:
            return np.pad(f, pad_width=1, mode='constant', constant_values=0)
        elif self.boundary_condition == BoundaryCondition.NEUMANN:
            return np.pad(f, pad_width=1, mode='edge')
        elif self.boundary_condition == BoundaryCondition.PERIODIC:
            padded = np.zeros((f.shape[0] + 2, f.shape[1] + 2))
            padded[1:-1, 1:-1] = f
            padded[0, 1:-1] = f[-1, :]
            padded[-1, 1:-1] = f[0, :]
            padded[1:-1, 0] = f[:, -1]
            padded[1:-1, -1] = f[:, 0]
            padded[0, 0] = f[-1, -1]
            padded[0, -1] = f[-1, 0]
            padded[-1, 0] = f[0, -1]
            padded[-1, -1] = f[0, 0]
            return padded
        else:  # DIRICHLET_COPY
            return np.pad(f, pad_width=1, mode='edge')

    @staticmethod
    def _shrink2(x: np.ndarray, y: np.ndarray, lambda_: float) -> Tuple[np.ndarray, np.ndarray]:
        """Soft shrinkage operator."""
        norm = np.sqrt(x**2 + y**2)
        norm[norm == 0] = lambda_
        magnitude = np.maximum(norm - lambda_, 0) / norm
        return x * magnitude, y * magnitude

    @staticmethod
    def _compute_derivatives(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward differences."""
        dx = np.zeros_like(u)
        dy = np.zeros_like(u)
        dx[:-1, :] = u[1:, :] - u[:-1, :]
        dy[:, :-1] = u[:, 1:] - u[:, :-1]
        return dx, dy

    @staticmethod
    def _compute_derivatives_transpose(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Compute backward differences (transpose of forward differences)."""
        return (-np.vstack((px[:-1, :], np.zeros((1, px.shape[1])))) + px + 
                -np.hstack((py[:, :-1], np.zeros((py.shape[0], 1)))) + py)

    def _denoise_channel(self, channel_data: np.ndarray) -> np.ndarray:
        """Denoise a single channel using Split Bregman iteration."""
        # Pad the input
        f_pad = self._pad_with_boundary_conditions(channel_data)
        u = f_pad.copy()
        
        # Initialize variables
        dx = np.zeros_like(u)
        dy = np.zeros_like(u)
        bx = np.zeros_like(u)
        by = np.zeros_like(u)
        
        for _ in range(self.max_iters):
            u_old = u.copy()
            
            # u-subproblem
            nabla_d = self._compute_derivatives_transpose(dx - bx, dy - by)
            rhs = self.mu * f_pad + self.gamma * nabla_d
            
            # Update u using Gauss-Seidel iterations
            for _ in range(2):
                u[1:-1, 1:-1] = (rhs[1:-1, 1:-1] + 
                                self.gamma * (u[2:, 1:-1] + u[:-2, 1:-1] + 
                                            u[1:-1, 2:] + u[1:-1, :-2])) / (self.mu + 4*self.gamma)
            
            # d-subproblem
            nabla_u_x, nabla_u_y = self._compute_derivatives(u)
            dx, dy = self._shrink2(nabla_u_x + bx, nabla_u_y + by, self.lambda_/self.gamma)
            
            # Update Bregman variables
            bx = bx + (nabla_u_x - dx)
            by = by + (nabla_u_y - dy)
            
            # Check convergence
            rel_error = np.linalg.norm(u - u_old) / np.linalg.norm(u)
            if rel_error < 1e-4:
                break
                
        return u[1:-1, 1:-1]

    def denoise(self, image: Image) -> Image:
        """Denoise an image using Split Bregman TV denoising."""
        if not image.is_color() and not image.is_grayscale():
            raise ValueError("Image must be color or grayscale")
            
        if image.is_color():
            denoised_data = np.zeros_like(image.data)
            for channel in range(image.shape[2]):
                denoised_data[..., channel] = self._denoise_channel(
                    image.data[..., channel]
                )
        else:
            denoised_data = self._denoise_channel(image.data)
            
        return Image(denoised_data, normalize=False)
    
# TODO: implement NonLocalMeansDenoiser | Another popular denoising method
# TODO: implement BilateralFilter | Edge-preserving smoothing
# TODO: implement WaveletDenoiser | Wavelet-based denoising