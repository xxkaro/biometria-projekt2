import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve

class IrisRingDivider:
    def __init__(self, image, x_pupil, y_pupil, r_pupil, r_iris):
        if len(image.shape) == 3:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = image
        self.x_pupil = x_pupil
        self.y_pupil = y_pupil
        self.r_pupil = r_pupil
        self.r_iris = r_iris

    def apply_gabor_filter(self, image, frequency=0.3, theta=0):
        # Image dimensions
        img_height, img_width = image.shape
        
        # Parameters for vertical (y) and horizontal (x) kernels
        sigma_y = 1 / (2 * np.pi * frequency)  # Vertical sigma (Gaussian)
        sigma_x = 1  # Horizontal sigma, used for mean kernel

        # Vertical kernel: Gaussian filter (1/8th of image height)
        vertical_range = img_height // 8  # 1/8th of the height of the image
        y = np.linspace(-vertical_range, vertical_range, 2 * vertical_range + 1)
        y_kernel = np.exp(-0.5 * (y / sigma_y) ** 2)  # Gaussian in y-direction
        y_kernel /= np.sum(y_kernel)  # Normalize the kernel

        # Horizontal kernel: Mean filter (1/128th of image width)
        horizontal_range = img_width // 128  # 1/128th of the width of the image
        x_kernel = np.ones(horizontal_range) / horizontal_range  # Uniform filter for averaging
        x_kernel = x_kernel.reshape(1, -1)  # Make it a 1D row vector for convolving horizontally

        # Apply Gaussian filter in the vertical direction
        vert_filtered = convolve(image, y_kernel[:, np.newaxis])  # Convolve along vertical axis
        
        # Apply mean filter in the horizontal direction
        horiz_filtered = convolve(vert_filtered, x_kernel)  # Then convolve the result horizontally

        # Compute the real and imaginary parts of the Gabor filter
        # Real part (cosine-modulated Gaussian)
        real_part = np.exp(-0.5 * (y[:, np.newaxis] / sigma_y) ** 2) * np.cos(2 * np.pi * frequency * y[:, np.newaxis])

        # Imaginary part (sine-modulated Gaussian)
        imag_part = np.exp(-0.5 * (y[:, np.newaxis] / sigma_y) ** 2) * np.sin(2 * np.pi * frequency * y[:, np.newaxis])

        # Convolve with real and imaginary parts
        real_response = convolve(horiz_filtered, real_part)
        imag_response = convolve(horiz_filtered, imag_part)

        return real_response, imag_response



    def normalize_iris(self, radial_res=64, angular_res=512):
        output = np.zeros((radial_res, angular_res), dtype=np.float32)

        for i in range(radial_res):
            r = i / (radial_res - 1)
            ring_index = int(r * 8)  # determine ring

            dense_theta = np.linspace(0, 2 * np.pi, 2000)
            mask = np.ones_like(dense_theta, dtype=bool)

            if ring_index <= 3:
                angle_limit = np.deg2rad(15)
                block = (np.abs(dense_theta - np.pi/2) <= angle_limit)
                mask[block] = False
            elif ring_index in [4, 5]:
                angle_limit = np.deg2rad(33.5)
                block = ((np.abs(dense_theta - np.pi/2) <= angle_limit) |
                        (np.abs(dense_theta - 3*np.pi/2) <= angle_limit))
                mask[block] = False
            elif ring_index in [6, 7]:
                angle_limit = np.deg2rad(45)
                block = ((np.abs(dense_theta - np.pi/2) <= angle_limit) |
                        (np.abs(dense_theta - 3*np.pi/2) <= angle_limit))
                mask[block] = False

            valid_theta = dense_theta[mask]

            if valid_theta.size == 0:
                output[i, :] = 0  # or np.nan for clarity
                continue

            # Stretch valid angles to cover full range
            stretched_theta = np.interp(
                np.linspace(0, 1, angular_res),
                np.linspace(0, 1, valid_theta.size),
                valid_theta
            )

            radius = self.r_pupil + r * (self.r_iris - self.r_pupil)
            x = self.x_pupil + radius * np.cos(stretched_theta)
            y = self.y_pupil + radius * np.sin(stretched_theta)

            ring_values = cv2.remap(
                self.image,
                x.astype(np.float32).reshape(1, -1),
                y.astype(np.float32).reshape(1, -1),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )

            output[i, :] = ring_values.flatten()

        return output

    def display_normalized_iris(self, radial_res=64, angular_res=512):
        normalized = self.normalize_iris(radial_res, angular_res)

        plt.figure(figsize=(10, 3))
        plt.imshow(normalized, cmap='gray', aspect='auto')
        plt.title("Unwrapped Iris")
        plt.axis('off')
        plt.show()

    def compute_valid_angles_for_radius(self, r):
        i = int(r * 8)  # determine which ring we're in (0-7)
        theta = np.linspace(0, 2 * np.pi, 1000)  # dense enough for precision
        mask = np.ones_like(theta, dtype=bool)

        if i <= 3:
            angle_limit = np.deg2rad(15)
            block = (np.abs(theta - np.pi/2) <= angle_limit)
            mask[block] = False
        elif i in [4, 5]:
            angle_limit = np.deg2rad(33.5)
            block = ((np.abs(theta - np.pi/2) <= angle_limit) |
                    (np.abs(theta - 3*np.pi/2) <= angle_limit))
            mask[block] = False
        elif i in [6, 7]:
            angle_limit = np.deg2rad(45)
            block = ((np.abs(theta - np.pi/2) <= angle_limit) |
                    (np.abs(theta - 3*np.pi/2) <= angle_limit))
            mask[block] = False

        return theta[mask]  # valid angles only

    def create_iris_code(self, radial_res=64, angular_res=512):
        normalized = self.normalize_iris(radial_res, angular_res)

        # Apply Gabor filter (simple version, single frequency and angle)
        real_response, imag_response = self.apply_gabor_filter(normalized, frequency=0.22, theta=0)

        # Initialize flattened iris code with alternating bits
        iris_code = np.zeros((radial_res, angular_res * 2), dtype=np.uint8)

        # Create bit masks
        bit1 = (real_response > 0).astype(np.uint8)  # First bit
        bit2 = (imag_response > 0).astype(np.uint8)  # Second bit

        # Fill alternating positions
        iris_code[:, 0::2] = bit1  # Even indices: 0, 2, 4, ..., 2N-2
        iris_code[:, 1::2] = bit2  # Odd indices: 1, 3, 5, ..., 2N-1

        return iris_code

    def display_iris_code(self, radial_res=64, angular_res=512):
        iris_code = self.create_iris_code(radial_res, angular_res)  # shape: (radial_res, angular_res * 2)

        num_horizontal = 8    # Display rows
        num_vertical = 256    # Display columns

        # Create the display matrix to match 8x256
        display_image = np.zeros((num_horizontal, num_vertical), dtype=np.uint8)

        for i in range(num_horizontal):
            for j in range(num_vertical):
                # Map i, j to the original code coordinates
                # Correctly reflect the alternating bit logic
                total_positions = angular_res  # because each bit-pair is linked to one angular step
                bit_idx = j % 2  # even -> 0 (bit1), odd -> 1 (bit2)
                angular_position = j // 2     # which original angular step to use

                # Clamp: in case num_vertical is odd
                if angular_position >= angular_res:
                    angular_position = angular_res - 1

                # Map the vertical section to the radial resolution
                row_idx = int(i * radial_res / num_horizontal)

                # Calculate the proper column in the flattened iris_code
                col_idx = angular_position * 2 + bit_idx

                display_image[i, j] = iris_code[row_idx, col_idx]

        # Plot as a black & white image
        plt.figure(figsize=(12, 3))
        plt.imshow(display_image, cmap='gray', aspect='auto')
        plt.title("Iris Code Visualization (First, Second, First, Second Bits)")
        plt.axis('off')
        plt.show()



    def calculate_hamming_distance(self, code1, code2):
        if code1.shape != code2.shape:
            raise ValueError("Iris codes must have the same shape for Hamming distance calculation.")
        return np.sum(code1 != code2) / code1.size

