import numpy as np
import cv2
import matplotlib.pyplot as plt

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

    def normalize_iris(self, radial_res=64, angular_res=512):
        theta = np.linspace(0, 2 * np.pi, angular_res)
        r = np.linspace(0, 1, radial_res)

        r_grid, theta_grid = np.meshgrid(r, theta)

        radius_diff = self.r_iris - self.r_pupil
        r_actual = self.r_pupil + r_grid.T * radius_diff

        x = self.x_pupil + r_actual * np.cos(theta_grid.T)
        y = self.y_pupil + r_actual * np.sin(theta_grid.T)

        normalized = cv2.remap(
            self.image,
            x.astype(np.float32),
            y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        return normalized

    def display_normalized_iris(self, radial_res=64, angular_res=512):
        normalized = self.normalize_iris(radial_res, angular_res)
        mask = self.create_normalized_ring_mask(radial_res, angular_res)

        masked_iris = np.where(mask == 1, normalized, 0)

        plt.figure(figsize=(10, 3))
        plt.imshow(masked_iris, cmap='gray', aspect='auto')
        plt.title("Unwrapped Iris (Masked)")
        plt.axis('off')
        plt.show()

    def gabor_kernel(self, frequency=0.25, theta=0, sigma=2.0):
        ksize = 31
        lambd = 1 / frequency
        gamma = 1.0
        psi = 0
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        return kernel

    def extract_iris_code(self, normalized_iris, mask, num_samples_per_ring=128, num_rings=8):
        rows, cols = normalized_iris.shape
        iris_code = np.zeros((num_rings, num_samples_per_ring), dtype=np.uint8)

        ring_height = rows // num_rings
        gabor = self.gabor_kernel(frequency=0.25)

        for ring_idx in range(num_rings):
            start_row = ring_idx * ring_height
            end_row = (ring_idx + 1) * ring_height if ring_idx != num_rings - 1 else rows

            ring_pixels = normalized_iris[start_row:end_row, :].flatten()
            ring_mask = mask[start_row:end_row, :].flatten()

            valid_pixels = ring_pixels[ring_mask == 1]

            if len(valid_pixels) == 0:
                continue

            sample_indices = np.linspace(0, len(valid_pixels) - 1, num_samples_per_ring, dtype=int)

            for bit_idx, idx in enumerate(sample_indices):
                pixel_value = valid_pixels[idx]
                response_real = np.real(np.exp(1j * pixel_value * gabor[15, 15]))
                response_imag = np.imag(np.exp(1j * pixel_value * gabor[15, 15]))
                phase = np.arctan2(response_imag, response_real)

                iris_code[ring_idx, bit_idx] = 1 if phase >= 0 else 0

        return iris_code

    def create_iris_code(self):
        normalized = self.normalize_iris(radial_res=64, angular_res=512)
        mask = self.create_normalized_ring_mask(radial_res=64, angular_res=512)
        iris_code = self.extract_iris_code(normalized, mask, num_samples_per_ring=128, num_rings=8)
        return iris_code

    def display_iris_code(self, iris_code):
        plt.figure(figsize=(10, 2))
        plt.imshow(iris_code, cmap='gray', aspect='auto')
        plt.title('Iris Code: 8 Rings × 128 Bits')
        plt.xlabel('Bit Index')
        plt.ylabel('Ring Index')
        plt.axis('off')
        plt.show()

    def create_normalized_ring_mask(self, radial_res=64, angular_res=512):
        theta = np.linspace(0, 2 * np.pi, angular_res)
        r = np.linspace(0, 1, radial_res)

        r_grid, theta_grid = np.meshgrid(r, theta)
        mask = np.ones_like(r_grid, dtype=np.uint8)  # start with all valid (1)

        for i in range(8):  # assuming 8 rings
            r_min = i / 8
            r_max = (i + 1) / 8

            in_ring = (r_grid >= r_min) & (r_grid <= r_max)

            angle_limit = 0

            if i <= 3:
                angle_limit = np.deg2rad(15)
                block = ((theta_grid >= (np.pi / 2 - angle_limit)) & (theta_grid <= (np.pi / 2 + angle_limit)))
                mask[block & in_ring] = 0

            elif i in [4, 5]:
                angle_limit = np.deg2rad(33.5)
                block = (((theta_grid >= (np.pi / 2 - angle_limit)) & (theta_grid <= (np.pi / 2 + angle_limit))) |
                         ((theta_grid >= (3 * np.pi / 2 - angle_limit)) & (theta_grid <= (3 * np.pi / 2 + angle_limit))))
                mask[block & in_ring] = 0

            elif i in [6, 7]:
                angle_limit = np.deg2rad(45)
                block = (((theta_grid >= (np.pi / 2 - angle_limit)) & (theta_grid <= (np.pi / 2 + angle_limit))) |
                         ((theta_grid >= (3 * np.pi / 2 - angle_limit)) & (theta_grid <= (3 * np.pi / 2 + angle_limit))))
                mask[block & in_ring] = 0

        return mask.T  # match the shape of the normalized image

    def calculate_hamming_distance(self, iris_code1, iris_code2):
        if iris_code1.shape != iris_code2.shape:
            raise ValueError("Iris codes must have the same shape to calculate Hamming distance.")

        difference = np.bitwise_xor(iris_code1, iris_code2)
        distance = np.sum(difference) / difference.size
        return distance
