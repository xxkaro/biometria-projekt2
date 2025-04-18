from io import BytesIO
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import sys
import cv2
import numpy as np
from processor import ImageProcessor 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure    
import matplotlib.pyplot as plt


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig) 
        self.axes = fig.add_subplot(111)
        self.axes.set_title("")
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        # self.axes.set_frame_on(False) 
        self.axes.spines['top'].set_visible(False)  
        self.axes.spines['right'].set_visible(False)  
        self.axes.spines['left'].set_visible(False) 
        self.axes.spines['bottom'].set_visible(False)
        fig.tight_layout() 
        self.draw()
    
class ImageProcessorGUI(QWidget):  
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Iris Recognition")
        self.setGeometry(350, 200, 800, 600)
        self.image_path = None  
        self.image_processor = None  
        self.processed_image = None  

        self.big_layout = QHBoxLayout()
        # Menu bar
        self.create_menu()

        # Main layout
        self.main_layout = QVBoxLayout()

        # Image layout
        self.image_layout = QHBoxLayout()
        self.image2_layout = QHBoxLayout()

        # Original image
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(400, 400)
        self.original_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.original_label.setStyleSheet("border: 2px solid black;")

        # Processed image
        self.processed_label = QLabel("Modified Image")
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_label.setMinimumSize(400, 400)
        self.processed_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.processed_label.setStyleSheet("border: 2px solid black;")

        self.unwrapped_label = QLabel("Unwraped iris")
        self.unwrapped_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.unwrapped_label.setMinimumSize(300, 50) 
        self.unwrapped_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.unwrapped_label.setStyleSheet("border: 2px solid black;")

        self.code_label = QLabel("Iris code")
        self.code_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.code_label.setMinimumSize(500, 50)
        self.code_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.code_label.setStyleSheet("border: 2px solid black;")

        # Add labels to layout
        self.image_layout.addWidget(self.original_label)
        self.image_layout.addWidget(self.processed_label)
        self.image2_layout.addWidget(self.unwrapped_label)
        self.image2_layout.addWidget(self.code_label)

        # Buttons layout
        self.button_layout = QVBoxLayout()

        # Morphological Operations
        self.morphological_operations_layout = QHBoxLayout()

        self.pupil_button = QPushButton("Detect pupil")
        self.pupil_button.clicked.connect(self.detect_pupil)
        self.pupil_button.setStyleSheet("background-color: #d5006d; color: white;") 
        self.morphological_operations_layout.addWidget(self.pupil_button)

        self.iris_button = QPushButton("Detect iris")
        self.iris_button.clicked.connect(self.detect_iris)
        self.iris_button.setStyleSheet("background-color: #d5006d; color: white;") 
        self.morphological_operations_layout.addWidget(self.iris_button)

        self.unwrap_button = QPushButton("Unwrap Iris")
        self.unwrap_button.clicked.connect(self.unwrap_iris)
        self.unwrap_button.setStyleSheet("background-color: #d5006d; color: white;")
        self.morphological_operations_layout.addWidget(self.unwrap_button)

        self.code_button = QPushButton("Iris Code")
        self.code_button.clicked.connect(self.iris_code)
        self.code_button.setStyleSheet("background-color: #d5006d; color: white;")
        self.morphological_operations_layout.addWidget(self.code_button)

        self.compare_button = QPushButton("Compare Iris Codes")
        self.compare_button.clicked.connect(self.compare_iris_codes)
        self.compare_button.setStyleSheet("background-color: #d5006d; color: white;")
        self.morphological_operations_layout.addWidget(self.compare_button)

        # Buttons layout
        self.button_layout2 = QVBoxLayout()

    
        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_image)
        self.button_layout2.addWidget(self.reset_button)

        # Layouts
        self.main_layout.addLayout(self.image_layout)
        self.main_layout.addLayout(self.image2_layout)
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addLayout(self.morphological_operations_layout)
        self.main_layout.addLayout(self.button_layout2)

        self.big_layout.addLayout(self.main_layout)

        self.setLayout(self.big_layout)

    def create_menu(self):
        menu_bar = QMenuBar(self)

        # Load menu
        load_menu = menu_bar.addMenu("Load")
        load_action = QAction("Load Image", self)
        load_action.triggered.connect(self.load_image)
        load_menu.addAction(load_action)

        # Edit menu with Undo/Redo
        edit_menu = menu_bar.addMenu("Edit")

        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut("Ctrl+Z") 
        self.undo_action.triggered.connect(self.undo)
        self.undo_action.setEnabled(False)
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut("Ctrl+Y") 
        self.redo_action.triggered.connect(self.redo)
        self.redo_action.setEnabled(False)
        edit_menu.addAction(self.redo_action)

        self.big_layout.setMenuBar(menu_bar)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif)")
        if file_path:
            self.image_path = file_path
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.original_image = img.copy() 
            self.image_processor = ImageProcessor(img)

            self.display_image(self.original_image, self.original_label)
            self.display_image(self.image_processor.image, self.processed_label)


    def display_image(self, image, label):
        """
        Display the image on the label in the GUI.
        """
        if image is None:
            return

        height, width = image.shape[:2]
        if len(image.shape) == 2:  
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 
            height, width = image.shape[:2]

        bytes_per_line = width * 3 
        image_data = image.tobytes()

        q_img = QImage(image_data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)

        pixmap = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)


    def detect_pupil(self):
        """Apply morphological opening to detect pupil."""
        if self.image_processor:
            self.processed_image, x, y, r = self.image_processor.detect_pupil()
            image = self.processed_image
            if image.dtype == bool or image.max() <= 1:
                image = (image * 255).astype(np.uint8)

            # 2. Jeśli obraz jest jednokanałowy (grayscale), konwertuj do BGR
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                display_image = image.copy()
            original_image = self.original_image.copy()
            # Rysujemy okrąg: (image, center, radius, color, thickness)
            cv2.circle(original_image, (int(x), int(y)), int(r), (0, 255, 0), 2)  # zielony okrąg
            self.display_image(display_image, self.processed_label)
            self.display_image(original_image, self.original_label)


    def detect_iris(self):  
        """Apply morphological closing to detect iris."""
        if self.image_processor:
            self.processed_image, x, y, r = self.image_processor.detect_iris()
            image = self.processed_image
            if image.dtype == bool or image.max() <= 1:
                image = (image * 255).astype(np.uint8)

            # 2. Jeśli obraz jest jednokanałowy (grayscale), konwertuj do BGR
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                display_image = image.copy()
            # Rysujemy okrąg: (image, center, radius, color, thickness)
            original_image = self.original_image.copy()
            cv2.circle(original_image, (int(x), int(y)), int(r), (0, 255, 0), 2)  # zielony okrąg
            self.display_image(display_image, self.processed_label)
            self.display_image(original_image, self.original_label)
            
            
    def unwrap_iris(self):
        """Unwrap the iris image."""
        if self.image_processor:    
            self.processed_image, x, y, r = self.image_processor.detect_iris()
            img, x_pupil, y_pupil, r_pupil = self.image_processor.detect_pupil()

            # Weź aktualny obraz (np. grayscale)
            image = self.image_processor.image

            unwrapped = self.image_processor.unwrap_iris(image, (x, y), r_pupil, r)

            display_image = unwrapped.copy()

            # Wyświetl na nowym labelu (upewnij się, że go masz!)
            self.display_image(display_image, self.unwrapped_label)


    def iris_code(self):
        """Unwrap the iris image and show as a matplotlib plot in QLabel."""
        if self.image_processor:    
            iris_code = self.image_processor.rings_division()

            # Create a figure and axes for displaying iris code image
            fig, ax = plt.subplots(figsize=(8, 1))  # Adjust size to fit QLabel
            self.display_iris_code_in_gui(ax, iris_code)  # Use display_iris_code_in_gui to generate plot
            ax.axis('off')  # Hide axes for clean display
            fig.tight_layout(pad=0)

            # Save the plot to a buffer
            buf = BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)

            # Convert buffer to QImage and then to QPixmap
            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)

            # Set the pixmap to QLabel
            self.code_label.setPixmap(pixmap.scaled(self.code_label.size(), 
                                                    Qt.AspectRatioMode.KeepAspectRatio,
                                                    Qt.TransformationMode.SmoothTransformation))

    
    def compare_iris_codes(self):
        """Compare the iris codes of two images."""
        if self.image_processor:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif)")
            if file_path:
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.display_image(img, self.processed_label)

                # Generate iris codes using the updated intertwined version
                iris_code1 = self.image_processor.rings_division()
                iris_code2 = self.image_processor.rings_division(img)

                # Calculate Hamming distance
                hamming_distance = self.image_processor.calculate_hamming_distance(iris_code1, iris_code2)

                # Create a dialog to display the result
                dialog = QDialog(self)
                dialog.setWindowTitle("Iris Code Comparison")
                layout = QVBoxLayout(dialog)

                # Create a Matplotlib figure for displaying iris codes
                fig = Figure(figsize=(12, 6))  # Set a larger figure size for better clarity
                canvas = FigureCanvas(fig)
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2)

                # Create images of iris codes like the first function
                self.display_iris_code_in_gui(ax1, iris_code1)
                ax1.set_title("Iris Code - Eye 1")
                ax1.axis('off')

                self.display_iris_code_in_gui(ax2, iris_code2)
                ax2.set_title("Iris Code - Eye 2")
                ax2.axis('off')

                layout.addWidget(canvas)

                # Add the Hamming distance label
                if hamming_distance < 0.22:
                    label = QLabel(f"Hamming distance: {hamming_distance:.2f} — Iris codes match.")
                else:
                    label = QLabel(f"Hamming distance: {hamming_distance:.2f} — Iris codes do not match.")
                layout.addWidget(label)

                dialog.setLayout(layout)
                dialog.exec()

    def display_iris_code_in_gui(self, ax, iris_code):
        """Displays the iris code (8x256) using Matplotlib on a given axis."""
        # Remove the last dimension if present (reshape from (8, 256, 1) to (8, 256))
        if iris_code.ndim == 3 and iris_code.shape[2] == 1:
            iris_code = iris_code.squeeze(-1)

        # Plot the iris code image on the provided axis
        ax.imshow(iris_code, cmap='gray', aspect='auto')
        ax.set_title("Iris Code")
        ax.axis('off')




    
    def undo(self):
        """Revert to the previous image."""
        if self.previous_image is not None:
            self.next_image = self.image_processor.image.copy()
            self.image_processor.image = self.previous_image.copy()
            self.processed_image = self.previous_image.copy()

            self.display_image(self.image_processor.image, self.processed_label)
            self.display_image(self.image_processor.image, self.original_label)

            self.redo_action.setEnabled(True)
            self.undo_action.setEnabled(False)

    def redo(self):
        """Reapply the undone change."""
        if self.next_image is not None:
            self.previous_image = self.image_processor.image.copy()
            self.image_processor.image = self.next_image.copy()
            self.processed_image = self.next_image.copy()

            self.display_image(self.image_processor.image, self.processed_label)
            self.display_image(self.image_processor.image, self.original_label)

            self.undo_action.setEnabled(True)  
            self.redo_action.setEnabled(False)

    def reset_image(self):
        """Reset the image to the original version."""
        if hasattr(self, 'original_image'):
            self.image_processor.image = self.original_image.copy()
            self.processed_image = self.original_image.copy()

            self.display_image(self.original_image, self.processed_label)
            self.display_image(self.original_image, self.original_label)
            empty_pixmap = QPixmap()
            self.unwrapped_label.setPixmap(empty_pixmap)
            self.code_label.setPixmap(empty_pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ImageProcessorGUI()
    window.show()
    sys.exit(app.exec())