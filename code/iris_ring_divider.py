import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve1d


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
        """
        Normalizes the iris image by unwrapping it into a rectangular format.
        The image is divided into rings based on the radial resolution and angular resolution.
        Each ring is then stretched to cover the full angular range.
        The function returns a 2D array representing the unwrapped iris.
        """
        output = np.zeros((radial_res, angular_res), dtype=np.float32)

        for i in range(radial_res):
            r = i / (radial_res - 1)
            ring_index = int(r * 8)  

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
                output[i, :] = 0 
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
    

    def compute_valid_angles_for_radius(self, r):
        """
        Computes the valid angles for a given radius in the iris.
        The angles are determined based on the ring index, which is derived from the radius.
        The function returns the angles that are not blocked by the pupil or other rings.
        """
        i = int(r * 8) 
        theta = np.linspace(0, 2 * np.pi, 1000)  
        mask = np.ones_like(theta, dtype=bool)

        if i <= 3:
            angle_limit = np.deg2rad(15)
            block = (np.abs(theta - np.pi/2) <= angle_limit)
            mask[block] = False
        elif i in [4, 5]:
            angle_limit = np.deg2rad(35)
            block = ((np.abs(theta - np.pi/2) <= angle_limit) |
                    (np.abs(theta - 3*np.pi/2) <= angle_limit))
            mask[block] = False
        elif i in [6, 7]:
            angle_limit = np.deg2rad(45)
            block = ((np.abs(theta - np.pi/2) <= angle_limit) |
                    (np.abs(theta - 3*np.pi/2) <= angle_limit))
            mask[block] = False

        return theta[mask] 
    

    def gabor_filter_1d(self, signal, frequency=0.3, sigma=3):
            length = len(signal)
            x = np.arange(-length // 2, length // 2)
            gabor_real = np.exp(-x**2 / (sigma**2)) * np.cos(2 * np.pi * frequency * x)
            gabor_imag = np.exp(-x**2 / (sigma**2)) * np.sin(2 * np.pi * frequency * x)
            gabor_real -= np.mean(gabor_real)
            real_response = convolve1d(signal, gabor_real, mode='reflect')
            imag_response = convolve1d(signal, gabor_imag, mode='reflect')
            return real_response, imag_response
    

    def create_iris_code(self, radial_res=64, angular_res=512, samples=128, gabor_freq=0.35, gabor_sigma=3):
        normalized = self.normalize_iris(radial_res, angular_res)
        num_stripes = 8
        gabor_sigma = 0.5 * gabor_freq * np.pi

        stripes = np.array_split(normalized, num_stripes, axis=0)
        iris_code = []

        for stripe in stripes:
            smoothed = gaussian_filter(stripe, sigma=2.5)
            signal = smoothed.mean(axis=0)  
            
            real_response, imag_response = self.gabor_filter_1d(signal, frequency=gabor_freq, sigma=gabor_sigma)

            segment_length = angular_res // samples
            interleaved_code = []

            for i in range(samples):
                start = i * segment_length
                end = (i + 1) * segment_length if i < samples - 1 else angular_res

                real_mean = real_response[start:end].mean()
                imag_mean = imag_response[start:end].mean()

                real_bit = int(real_mean > 0)
                imag_bit = int(imag_mean > 0)
                interleaved_code.extend([real_bit, imag_bit])  

            iris_code.append(interleaved_code) 

        iris_code_array = np.array(iris_code, dtype=np.uint8).reshape((8, 256, 1))

        return iris_code_array


    def display_iris_code(self):
        """
        Visualizes the 8 x 256 x 1 iris code as a grayscale image.
        Each stripe becomes a row, and interleaved bits are displayed as white (1) or black (0).
        """
        iris_code_array = self.create_iris_code()
        display_image = iris_code_array.squeeze(axis=2) 

        plt.figure(figsize=(12, 3))
        plt.imshow(display_image, cmap='gray', aspect='auto')
        plt.title("Iris Code Visualization (First, Second, First, Second Bits)")
        plt.axis('off')
        plt.show()


    def display_normalized_iris(self, radial_res=64, angular_res=512):
        """
        Visualizes the normalized iris image.
        The image is displayed in grayscale, with the radial resolution determining the number of rows
        and the angular resolution determining the number of columns.
        """
        normalized = self.normalize_iris(radial_res, angular_res)

        plt.figure(figsize=(10, 3))
        plt.imshow(normalized, cmap='gray', aspect='auto')
        plt.title("Unwrapped Iris")
        plt.axis('off')
        plt.show()


    def calculate_hamming_distance(self, code1, code2):
        """
        Calculates the Hamming distance between two iris codes.
        """
        if code1.shape != code2.shape:
            raise ValueError("Iris codes must have the same shape for Hamming distance calculation.")
        return np.sum(code1 != code2) / code1.size


