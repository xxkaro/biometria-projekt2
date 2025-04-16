import numpy as np
import matplotlib.pyplot as plt
import cv2
from iris_ring_divider import IrisRingDivider


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

    def detect_pupil(self, image=None, threshold_pupil=4.5):
        """
        Detect the pupil in the image using a simple thresholding method and OpenCV morphology.
        """
        if image is None:
            image = self.image

        binazation_threshold = self.calculate_binarization_threshold()
        threshold = binazation_threshold / threshold_pupil

        pupil_image = self.binarization(threshold).astype(np.uint8)


        # Krok 2: Erozja (usuwanie nadmiaru blasków – to pomoże w usunięciu jasnych punktów)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Kwadratowy kernel 3x3
        pupil_image = cv2.erode(pupil_image, kernel_erode)
                
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Kwadratowy kernel 3x3
        pupil_image = cv2.erode(pupil_image, kernel_erode)

        kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))  # Kwadratowy kernel 3x3
        pupil_image = cv2.morphologyEx(pupil_image, cv2.MORPH_OPEN, kernel_opening)

        # Krok 3: Dylatacja (przywrócenie krawędzi źrenicy bez rozlewania blasków)
        for i in range(3):
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Kwadratowy kernel 3x3
            pupil_image = cv2.dilate(pupil_image, kernel_dilate)

        for i in range(150):
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Kwadratowy kernel 5x5
            pupil_image = cv2.morphologyEx(pupil_image, cv2.MORPH_CLOSE, kernel_close)

        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Kwadratowy kernel 3x3
        pupil_image = cv2.dilate(pupil_image, kernel_erode)

        
        vertical_projection = np.sum(pupil_image, axis=0) // 255
        horizontal_projection = np.sum(pupil_image, axis=1) // 255
        diagonal_projection_left, diagonal_projection_right = self.diagonal_projection() 
        diagonal_projection_left = np.sum(diagonal_projection_left) // 255
        diagonal_projection_right = np.sum(diagonal_projection_right) // 255

        argmins_x = np.argpartition(vertical_projection, 5)[:5]
        argmins_y = np.argpartition(horizontal_projection, 5)[:5]

        # Posortuj te indeksy według odpowiadających im wartości
        argmins_x_sorted = argmins_x[np.argsort(vertical_projection[argmins_x])]
        argmins_y_sorted = argmins_y[np.argsort(horizontal_projection[argmins_y])]

        # Wybierz środkowy (3. najmniejszy z 5)
        x = argmins_x_sorted[2]
        y = argmins_y_sorted[2]

        # x, y = np.argmin(vertical_projection), np.argmin(horizontal_projection)
        r_vertical = np.max(vertical_projection) - np.min(vertical_projection)
        r_horizontal = np.max(horizontal_projection) - np.min(horizontal_projection)
        # r = np.mean([r_vertical, r_horizontal, diagonal_projection_left, diagonal_projection_right]) // 2
        r = np.mean([r_vertical, r_horizontal]) // 2
        r = r_vertical // 2

        return pupil_image, x, y, r

    
    def detect_iris(self, image=None, threshold_iris=1.3):
        """
        Detect the iris in the image using a simple thresholding method and OpenCV morphology.
        """
        if image is None:
            image = self.image

        binazation_threshold = self.calculate_binarization_threshold()
        threshold = binazation_threshold / threshold_iris

        iris_image = self.binarization(threshold).astype(np.uint8)

        # Krok 1: Delikatny opening – usuwanie rzęs bez wpływu na tęczówkę
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # mniejszy kernel
        iris_image = cv2.dilate(iris_image, kernel_dilate)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # mniejszy kernel
        iris_image = cv2.morphologyEx(iris_image, cv2.MORPH_OPEN, kernel_open)

        # Krok 2: Mniej agresywny closing – domykanie konturów tęczówki
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # zmniejszenie rozmiaru kernela
        iris_image = cv2.morphologyEx(iris_image, cv2.MORPH_CLOSE, kernel_close)

        # Krok 3: Delikatna dylatacja – przywracanie krawędzi, ale bez rozlewania
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # mniejszy kernel
        iris_image = cv2.dilate(iris_image, kernel_dilate)

        for i in range(10):
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Kwadratowy kernel 5x5
            iris_image = cv2.morphologyEx(iris_image, cv2.MORPH_CLOSE, kernel_close)


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
        h, w, c = self.image.shape

        # Konwersja do grayscale (prosty sposób: średnia z kanałów)
        gray = np.mean(self.image, axis=2).astype(np.uint8)

        # ↘ kierunek (i - j)
        main_diag_proj = np.array([np.sum(np.diag(gray, k)) for k in range(-h + 1, w)]) // 255

        # ↙ kierunek (i + j)
        flipped = np.flipud(gray)
        anti_diag_proj = np.array([np.sum(np.diag(flipped, k)) for k in range(-h + 1, w)]) // 255

        return main_diag_proj, anti_diag_proj

    def unwrap_iris(self, image, center, pupil_radius, iris_radius, height=30, width=120):
        """
        Unwrap the iris region from polar to Cartesian coordinates,
        excluding the pupil (starts from pupil_radius to iris_radius).
        """
        x_center, y_center = center

        r = np.linspace(pupil_radius, iris_radius, height)
        theta = np.linspace(0, 2 * np.pi, width)

        theta_grid, r_grid = np.meshgrid(theta, r)

        x = x_center + r_grid * np.cos(theta_grid)
        y = y_center + r_grid * np.sin(theta_grid)

        map_x = x.astype(np.float32)
        map_y = y.astype(np.float32)

        if len(image.shape) == 3 and image.shape[2] == 3:
            channels = [cv2.remap(image[:, :, c], map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        for c in range(3)]
            unwrapped = cv2.merge(channels)
        else:
            unwrapped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        h = unwrapped.shape[0]
        crop_margin = int(h * 0.02)
        unwrapped = unwrapped[crop_margin : h - crop_margin, :]

        return unwrapped

        return unwrapped

    def rings_division(self, image=None):
        """
        Divide the iris into rings and sectors.
        """
        x_pupil, y_pupil, r_pupil = self.detect_pupil()[1:]
        r_iris = self.detect_iris()[3]

        if image is None:
            image = self.image

        divider = IrisRingDivider(image, x_pupil, y_pupil, r_pupil, r_iris)
        
        iris_code = divider.create_iris_code()

        #divider.display_normalized_iris()
        #divider.display_iris_code(iris_code)

        return iris_code

    def calculate_hamming_distance(self, iris_code1, iris_code2):
        if iris_code1.shape != iris_code2.shape:
            raise ValueError("Iris codes must have the same shape to calculate Hamming distance.")

        difference = np.bitwise_xor(iris_code1, iris_code2)
        distance = np.sum(difference) / difference.size
        return distance
