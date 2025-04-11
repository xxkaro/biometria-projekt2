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

    def detect_pupil(self, image=None, threshold_pupil=3.9):
        """
        Detect the pupil in the image using a simple thresholding method and OpenCV morphology.
        """
        if image is None:
            image = self.image

        binazation_threshold = self.calculate_binarization_threshold()
        threshold = binazation_threshold / threshold_pupil

        pupil_image = self.binarization(threshold).astype(np.uint8)

        # Krok 1: Mocne closing (z większym kernel'em, aby zamknąć artefakty zewnętrzne)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Kwadratowy kernel 5x5
        pupil_image = cv2.morphologyEx(pupil_image, cv2.MORPH_CLOSE, kernel_close)

        # Krok 2: Erozja (usuwanie nadmiaru blasków – to pomoże w usunięciu jasnych punktów)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Kwadratowy kernel 3x3
        pupil_image = cv2.erode(pupil_image, kernel_erode)

        # Krok 3: Dylatacja (przywrócenie krawędzi źrenicy bez rozlewania blasków)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Kwadratowy kernel 3x3
        pupil_image = cv2.dilate(pupil_image, kernel_dilate)

        # Krok 4: Zamknięcie linii w poziomie (usuwa poziome artefakty, np. cienkie linie rzęs)
        kernel_close_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # Element w poziomie
        pupil_image = cv2.morphologyEx(pupil_image, cv2.MORPH_CLOSE, kernel_close_horizontal)

        # Krok 5: Zamknięcie linii w pionie (usuwa pionowe artefakty, np. cienkie linie)
        kernel_close_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # Element w pionie
        pupil_image = cv2.morphologyEx(pupil_image, cv2.MORPH_CLOSE, kernel_close_vertical)

        
        vertical_projection = np.sum(pupil_image, axis=0) // 255
        horizontal_projection = np.sum(pupil_image, axis=1) // 255
        diagonal_projection_left, diagonal_projection_right = self.diagonal_projection() 
        diagonal_projection_left = np.sum(diagonal_projection_left) // 255
        diagonal_projection_right = np.sum(diagonal_projection_right) // 255

        x, y = np.argmin(vertical_projection), np.argmin(horizontal_projection)
        r_vertical = np.max(vertical_projection) - np.min(vertical_projection)
        r_horizontal = np.max(horizontal_projection) - np.min(horizontal_projection)
        r = np.mean([r_vertical, r_horizontal, diagonal_projection_left, diagonal_projection_right]) // 2
        r = r_vertical // 2

        return pupil_image, x, y, r

    
    def detect_iris(self, image=None, threshold_iris=1.45):
        """
        Detect the iris in the image using a simple thresholding method and OpenCV morphology.
        """
        if image is None:
            image = self.image

        binazation_threshold = self.calculate_binarization_threshold()
        threshold = binazation_threshold / threshold_iris

        iris_image = self.binarization(threshold).astype(np.uint8)

        # Krok 1: Delikatny opening – usuwanie rzęs bez wpływu na tęczówkę
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # mniejszy kernel
        iris_image = cv2.morphologyEx(iris_image, cv2.MORPH_OPEN, kernel_open)

        # Krok 2: Mniej agresywny closing – domykanie konturów tęczówki
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # zmniejszenie rozmiaru kernela
        iris_image = cv2.morphologyEx(iris_image, cv2.MORPH_CLOSE, kernel_close)

        # Krok 3: Delikatna dylatacja – przywracanie krawędzi, ale bez rozlewania
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # mniejszy kernel
        iris_image = cv2.dilate(iris_image, kernel_dilate)


        vertical_projection = np.sum(iris_image, axis=0) // 255
        horizontal_projection = np.sum(iris_image, axis=1) // 255
        diagonal_projection_left, diagonal_projection_right = self.diagonal_projection() 
        diagonal_projection_left = np.sum(diagonal_projection_left) // 255
        diagonal_projection_right = np.sum(diagonal_projection_right) // 255

        img, x, y, r = self.detect_pupil(iris_image)
        r_vertical = np.max(vertical_projection) - np.min(vertical_projection)
        r_horizontal = np.max(horizontal_projection) - np.min(horizontal_projection)
        r = np.mean([r_vertical, r_horizontal, diagonal_projection_left, diagonal_projection_right]) // 2
        r = r_vertical // 2

        return iris_image, x, y, r

    def diagonal_projection(self):
        # Zakładamy, że self.image to obraz RGB
        print(self.image.shape)
        h, w, c = self.image.shape

        # Konwersja do grayscale (prosty sposób: średnia z kanałów)
        gray = np.mean(self.image, axis=2).astype(np.uint8)

        # ↘ kierunek (i - j)
        main_diag_proj = np.array([np.sum(np.diag(gray, k)) for k in range(-h + 1, w)]) // 255

        # ↙ kierunek (i + j)
        flipped = np.flipud(gray)
        anti_diag_proj = np.array([np.sum(np.diag(flipped, k)) for k in range(-h + 1, w)]) // 255

        return main_diag_proj, anti_diag_proj
