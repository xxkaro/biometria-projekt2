import numpy as np
import matplotlib.pyplot as plt
import cv2


class ImageProcessor:
    def __init__(self, image):
        self.image = image

    def convert_to_gray(self):
        """
        Convert the image to grayscale.
        """
        gray = np.dot(self.image[..., :3], [0.114, 0.587, 0.299])
        gray = gray.astype(np.uint8)
        # Convert (H, W) → (H, W, 3) for GUI compatibility
        # gray = np.stack([gray] * 3, axis=-1)
        return gray

    def calculate_binarization_threshold(self):
        h, w = self.image.shape[:2]
        if len(self.image.shape) == 3:
            gray_image = self.convert_to_gray()
        else:
            gray_image = self.image

        P = np.sum(gray_image) / (h * w)

        return P
    
    def binarization(self, threshold=128):
        """
        Binarization of an image
        """
        gray = self.convert_to_gray()
        binary = np.where(gray > threshold, 255, 0)
        return binary.astype(np.uint8)

    def detect_pupil(self, image=None, threshold_pupil=8):
        """
        Detect the pupil in the image using a simple thresholding method and OpenCV morphology.
        """
        if image is None:
            image = self.image

        binazation_threshold = self.calculate_binarization_threshold()
        threshold = binazation_threshold / threshold_pupil

        pupil_image = self.binarization(threshold).astype(np.uint8)

        # krok 1: opening (usuwa cienkie struktury - rzęsy)
        pupil_image = cv2.morphologyEx(pupil_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # krok 2: closing (zamyka małe dziurki)
        pupil_image = cv2.morphologyEx(pupil_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # krok 3: erozja (zmniejsza/usuwa źrenicę)
        pupil_image = cv2.erode(pupil_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        # krok 4: (opcjonalnie) dylatacja, by przywrócić wielkość
        pupil_image = cv2.dilate(pupil_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        vertical_projection = np.sum(pupil_image, axis=0) // 255
        horizontal_projection = np.sum(pupil_image, axis=1) // 255

        x, y = np.argmin(vertical_projection), np.argmin(horizontal_projection)
        # r = np.mean([np.min(vertical_projection), np.min(horizontal_projection)]) / 2
        r_vertical = np.max(vertical_projection) - np.min(vertical_projection)
        r_horizontal = np.max(horizontal_projection) - np.min(horizontal_projection)
        r = np.mean([r_vertical, r_horizontal]) // 2

        return pupil_image, x, y, r

    
    def detect_iris(self, image=None, threshold_iris=3):
        """
        Detect the iris in the image using a simple thresholding method and OpenCV morphology.
        """
        if image is None:
            image = self.image

        binazation_threshold = self.calculate_binarization_threshold()
        threshold = binazation_threshold / threshold_iris

        iris_image = self.binarization(threshold).astype(np.uint8)

        # krok 1: opening (usuwa cienkie struktury - rzęsy)
        iris_image = cv2.morphologyEx(iris_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # krok 2: closing (zamyka małe dziurki)
        iris_image = cv2.morphologyEx(iris_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # krok 3: erozja (zmniejsza/usuwa źrenicę)
        iris_image = cv2.erode(iris_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        # krok 4: (opcjonalnie) dylatacja, by przywrócić wielkość
        iris_image = cv2.dilate(iris_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        vertical_projection = np.sum(iris_image, axis=0) // 255
        horizontal_projection = np.sum(iris_image, axis=1) // 255

        img, x, y, r = self.detect_pupil(iris_image)
        r_vertical = np.max(vertical_projection) - np.min(vertical_projection)
        r_horizontal = np.max(horizontal_projection) - np.min(horizontal_projection)
        r = np.mean([r_vertical, r_horizontal]) // 2

        return iris_image, x, y, r
