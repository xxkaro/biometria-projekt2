import numpy as np
import matplotlib.pyplot as plt


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
        gray = np.stack([gray] * 3, axis=-1)
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

        
    def opening(self, kernel=None):
        """
        Opening of an image
        """
        if kernel is None:
            kernel = self.get_kernel_bin("square")  

        eroded_img = self.convolute_binary(self.image, kernel_shape="square", operation="erosion")

        return self.convolute_binary(eroded_img, kernel_shape="square", operation="dilation") 

    def closing(self, kernel=None):  
        """
        Closing of an image
        """
        if kernel is None:
            kernel = self.get_kernel_bin("square") 

        dilated_img = self.convolute_binary(self.image, kernel_shape="square", operation="dilation")

        return self.convolute_binary(dilated_img, kernel_shape="square", operation="erosion")
    
    def get_kernel_bin(self, shape="square"):
        """Generate a kernel based on the selected shape."""
        if shape == "square":
            return np.zeros((3, 3), dtype=np.uint8)
        elif shape == "cross":
            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, :] = 0
            kernel[:, 1] = 0
            return kernel
        elif shape == "vertical_line":
            return np.array([[0, 1, 0],
                             [0, 1, 0],
                             [0, 1, 0]])
        elif shape == "horizontal_line":
            return np.array([[0, 0, 0],
                             [1, 1, 1],
                             [0, 0, 0]])
        else:
            raise ValueError(f"Unknown kernel shape: {shape}")

    def convolute_binary(self, image=None, kernel_shape="square", operation="dilation"):
        """
        Perform binary image convolution for either dilation or erosion depending on the operation parameter.
        
        Parameters:
            image (np.array): The binary image to process (default is the class image).
            kernel_shape (str): The shape of the kernel to use (e.g., "square", "cross", etc.).
            operation (str): The operation to perform: 'dilation' for dilation, 'erosion' for erosion.
            
        Returns:
            np.array: The processed image after applying the selected operation.
        """
        kernel = self.get_kernel_bin(kernel_shape)
        if image is None:
            image = self.image

        kernel_size = kernel.shape[0]
        pad = kernel_size // 2

        image_padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        if operation == "dilation":
            output = np.ones_like(image, dtype=np.uint8) * 255 
        elif operation == "erosion":
            output = np.ones_like(image, dtype=np.uint8) * 255
        else:
            raise ValueError(f"Invalid operation: {operation}. Use 'dilation' or 'erosion'.")
        output = np.pad(output, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = image_padded[i:i + kernel_size, j:j + kernel_size]

                if operation == "dilation":
                    if region[1, 1, 1] == 0:
                        output[i, j] = 0
                        for k in range(kernel_size):
                            for l in range(kernel_size):
                                if kernel[k, l] == 0:
                                    output[i + k, j + l] = 0

                elif operation == "erosion":
                    erosion_condition = True
                    for k in range(kernel_size):
                        for l in range(kernel_size):
                            if kernel[k, l] == 0 and region[1, k, l].any() != 0:  
                                erosion_condition = False
                                break
                        if not erosion_condition:
                            break

                    if not erosion_condition:
                        output[i, j] = 255
                    else:
                        output[i, j] = 0

        return output[pad:-pad, pad:-pad]
    

    def detect_pupil(self, image=None, threshold_pupil=8, kernel_shape="square"):
        """
        Detect the pupil in the image using a simple thresholding method.
        """
        if image is None:
            image = self.image
        
        binazation_threshold = self.calculate_binarization_threshold()
        threshold = binazation_threshold / threshold_pupil

        pupil_image = self.binarization(threshold)
        pupil_image = self.convolute_binary(pupil_image, kernel_shape="square", operation="erosion")
        # pupil_image = self.closing(pupil_image)
        # pupil_image = self.opening(pupil_image)

        vertical_projection = np.sum(pupil_image, axis=0)
        horizontal_projection = np.sum(pupil_image, axis=1)

        x, y = np.max(vertical_projection), np.max(horizontal_projection)

        return pupil_image, x, y
    
    def detect_iris(self, image=None, threshold_iris=4, kernel_shape="square"):
        """
        Detect the iris in the image using a simple thresholding method.
        """
        if image is None:
            image = self.image

        binazation_threshold = self.calculate_binarization_threshold()
        threshold = binazation_threshold / threshold_iris

        iris_image = self.binarization(threshold)
        iris_image = self.convolute_binary(iris_image, kernel_shape="square", operation="erosion")
        # iris_image = self.closing(iris_image)
        # iris_image = self.opening(iris_image)

        vertical_projection = np.sum(iris_image, axis=0)
        horizontal_projection = np.sum(iris_image, axis=1)

        x, y = np.max(vertical_projection), np.max(horizontal_projection)

        return iris_image, x, y

    