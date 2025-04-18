import numpy as np
import matplotlib.pyplot as plt
import cv2
from iris_ring_divider import IrisRingDivider


class ImageProcessor:
    def __init__(self, image):
        self.image = image
        

    def convert_to_gray(self, image=None):
        """
        Convert the image to grayscale.
        """
        if image is None:
            image = self.image
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # Convert (H, W) → (H, W, 3) for GUI compatibility
        # gray = np.stack([gray] * 3, axis=-1)
        return gray
    

    def calculate_binarization_threshold(self, image=None):
        """
        Calculate the binarization threshold for the image.
        """
        if image is None:
            image = self.image

        h, w = image.shape[:2]
        if len(image.shape) == 3:
            gray_image = self.convert_to_gray()
        else:
            gray_image = image

        P = np.sum(gray_image) / (h * w)

        return P
    
    
    def binarization(self, image=None, threshold=128):
        """
        Binarization of an image
        """
        if image is None:
            image = self.image

        gray = self.convert_to_gray(image)
        binary = np.where(gray > threshold, 255, 0)
        return binary.astype(np.uint8)
    

    def detect_pupil(self, image=None, threshold_pupil=4.1):
        """
        Detect the pupil in the image using a simple thresholding method and OpenCV morphology.
        """
        if image is None:
            image = self.image

        binazation_threshold = self.calculate_binarization_threshold(image)
        threshold = binazation_threshold / threshold_pupil

        pupil_image = self.binarization(image, threshold).astype(np.uint8)

        pupil_image = cv2.medianBlur(pupil_image, 3)

        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
        pupil_image = cv2.erode(pupil_image, kernel_erode)

        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
        pupil_image = cv2.erode(pupil_image, kernel_erode)

        kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1)) 
        pupil_image = cv2.morphologyEx(pupil_image, cv2.MORPH_OPEN, kernel_opening)

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
        pupil_image = cv2.dilate(pupil_image, kernel_dilate, iterations=3)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  
        pupil_image = cv2.morphologyEx(pupil_image, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 
        pupil_image = cv2.dilate(pupil_image, kernel_erode)

        vertical_projection = np.sum(pupil_image, axis=0) // 255
        horizontal_projection = np.sum(pupil_image, axis=1) // 255

        argmins_x = np.argpartition(vertical_projection, 7)[:7]
        argmins_y = np.argpartition(horizontal_projection, 7)[:7]

        argmins_x_sorted = argmins_x[np.argsort(vertical_projection[argmins_x])]
        argmins_y_sorted = argmins_y[np.argsort(horizontal_projection[argmins_y])]

        x = argmins_x_sorted[4]
        y = argmins_y_sorted[4]

        r_vertical = np.max(vertical_projection) - np.min(vertical_projection)
        r_horizontal = np.max(horizontal_projection) - np.min(horizontal_projection)

        r = np.mean([r_vertical, r_horizontal]) // 2

        return pupil_image, x, y, r

    
    def detect_iris(self, image=None, threshold_iris=1.3):
        """
        Detect the iris in the image using a simple thresholding method and OpenCV morphology.
        """
        if image is None:
            image = self.image

        binazation_threshold = self.calculate_binarization_threshold(image)
        threshold = binazation_threshold / threshold_iris

        iris_image = self.binarization(image, threshold).astype(np.uint8)
        iris_image = cv2.medianBlur(iris_image, 3)

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  
        iris_image = cv2.dilate(iris_image, kernel_dilate)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10)) 
        iris_image = cv2.morphologyEx(iris_image, cv2.MORPH_OPEN, kernel_open, iterations=2)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1)) 
        iris_image = cv2.morphologyEx(iris_image, cv2.MORPH_OPEN, kernel_open, iterations=2)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  
        iris_image = cv2.morphologyEx(iris_image, cv2.MORPH_CLOSE, kernel_close, iterations=11)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  
        iris_image = cv2.morphologyEx(iris_image, cv2.MORPH_OPEN, kernel_open)

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  
        iris_image = cv2.dilate(iris_image, kernel_dilate, iterations=3)

        img, x, y, r = self.detect_pupil(image)

        r = self.estimate_iris_radius_binary(iris_image, (x, y), r, white_threshold_ratio=0.45)

        return iris_image, x, y, r
    
    
    def estimate_iris_radius_binary(self, binary_image, center, r_pupil, max_r=200 , white_threshold_ratio=0.4):
        """
        Estimate the radius of the iris based on a binary image.
        """
        cx, cy = center
        h, w = binary_image.shape
        r_pupil = int(r_pupil)

        for r in range(r_pupil + 1, max_r):
            theta = np.linspace(0, 2 * np.pi, 360)
            x = (cx + r * np.cos(theta)).astype(int)
            y = (cy + r * np.sin(theta)).astype(int)

            valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            x, y = x[valid], y[valid]

            values = binary_image[y, x]
            white_ratio = np.sum(values == 255) / len(values)

            if white_ratio > white_threshold_ratio:
                return r

        return max_r


    def diagonal_projection(self):
        h, w, c = self.image.shape

        gray = np.mean(self.image, axis=2).astype(np.uint8)

        # ↘ kierunek (i - j)
        main_diag_proj = np.array([np.sum(np.diag(gray, k)) for k in range(-h + 1, w)]) // 255

        # ↙ kierunek (i + j)
        flipped = np.flipud(gray)
        anti_diag_proj = np.array([np.sum(np.diag(flipped, k)) for k in range(-h + 1, w)]) // 255

        return main_diag_proj, anti_diag_proj
    

    def unwrap_iris(self, image, height=30, width=120):
        x_pupil, y_pupil, r_pupil = self.detect_pupil()[1:]
        r_iris = self.detect_iris()[3]

        if image is None:
            image = self.image

        divider = IrisRingDivider(image, x_pupil, y_pupil, r_pupil, r_iris)

        output = divider.normalize_iris(radial_res=height, angular_res=width)

        output_min = np.min(output)
        output_max = np.max(output)
        output_normalized = 255 * (output - output_min) / (output_max - output_min)  

        output_image = output_normalized.astype(np.uint8)

        return output_image
    

    def rings_division(self, image=None):
        """
        Divide the iris into rings and sectors.
        """
        if image is None:
            image = self.image

        x_pupil, y_pupil, r_pupil = self.detect_pupil(image=image)[1:]
        r_iris = self.detect_iris(image=image)[3]

        divider = IrisRingDivider(image, x_pupil, y_pupil, r_pupil, r_iris)
        
        iris_code = divider.create_iris_code()

        return iris_code


    def calculate_hamming_distance(self, iris_code1, iris_code2):
        if iris_code1.shape != iris_code2.shape:
            raise ValueError("Iris codes must have the same shape to calculate Hamming distance.")

        return np.sum(iris_code1 != iris_code2) / iris_code1.size
