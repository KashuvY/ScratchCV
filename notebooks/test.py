# Example usage of Snake Segmentation

import importlib
import cv_lib
import cv_lib.segmentation.snake
importlib.reload(cv_lib.segmentation.snake)
from cv_lib.segmentation.snake import Snake, SnakeParams

# Load test image
image = cv_lib.Image("img/panda.jpg")
image = image.to_grayscale()

# Create snake with custom parameters
params = SnakeParams(
    alpha=0.01,  # Controls tension
    beta=0.1,   # Controls rigidity
    gamma=0.5,  # Step size
    max_iters=100,
    convergence_threshold=0.1
)
snake = Snake(image, params)

# Initialize snake interactively
print("Draw initial contour by clicking points. Press Enter when done.")
snake.initialize_interactive()

# Evolve snake with animation
final_contour = snake.evolve(animate=True)

# Save results
snake.save_result("segmented_with_contour.jpg")  # Full image with snake
snake.save_result("mask.jpg", mask_only=True)    # Binary mask only

# You can also get the mask as numpy array
mask = snake.get_mask()
print(f"Segmented region covers {mask.sum()/mask.size:.1%} of the image")