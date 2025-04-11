from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import sys
import cv2
import numpy as np
from processor import ImageProcessor 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

        self.setWindowTitle("Bildverarbeitung - Image Processing")
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

        # Original image
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setFixedSize(400, 400)
        self.original_label.setStyleSheet("border: 2px solid black;")

        # Processed image
        self.processed_label = QLabel("Modified Image")
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_label.setFixedSize(400, 400)
        self.processed_label.setStyleSheet("border: 2px solid black;")

        # Add labels to layout
        self.image_layout.addWidget(self.original_label)
        self.image_layout.addWidget(self.processed_label)


        # Buttons layout
        self.button_layout = QVBoxLayout()

        # Morphological Operations
        self.morphological_operations_layout = QHBoxLayout()
        #add dropdown to choose kernel shape


        self.pupil_button = QPushButton("Detect pupil")
        self.pupil_button.clicked.connect(self.detect_pupil)
        self.pupil_button.setStyleSheet("background-color: #d5006d; color: white;") 
        self.morphological_operations_layout.addWidget(self.pupil_button)

        self.iris_button = QPushButton("Detect iris")
        self.iris_button.clicked.connect(self.detect_iris)
        self.iris_button.setStyleSheet("background-color: #d5006d; color: white;") 
        self.morphological_operations_layout.addWidget(self.iris_button)

        # Buttons layout
        self.button_layout2 = QVBoxLayout()

        # Apply changes button
        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.clicked.connect(self.apply_changes)
        self.button_layout2.addWidget(self.apply_button)
    
        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_image)
        self.button_layout2.addWidget(self.reset_button)

        # Layouts
        self.main_layout.addLayout(self.image_layout)
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

        # Save menu
        save_menu = menu_bar.addMenu("Save")
        save_jpg_action = QAction("Save as JPG", self)
        save_jpg_action.triggered.connect(lambda: self.save_image("jpg"))
        save_menu.addAction(save_jpg_action)

        save_png_action = QAction("Save as PNG", self)
        save_png_action.triggered.connect(lambda: self.save_image("png"))
        save_menu.addAction(save_png_action)

        save_bmp_action = QAction("Save as BMP", self)
        save_bmp_action.triggered.connect(lambda: self.save_image("bmp"))
        save_menu.addAction(save_bmp_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(QApplication.instance().quit)
        save_menu.addAction(exit_action)

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
            # Rysujemy okrąg: (image, center, radius, color, thickness)
            cv2.circle(display_image, (int(x), int(y)), int(r), (0, 255, 0), 2)  # zielony okrąg
            self.display_image(display_image, self.processed_label)

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
            cv2.circle(display_image, (int(x), int(y)), int(r), (0, 255, 0), 2)  # zielony okrąg
            self.display_image(display_image, self.processed_label)

    
    def apply_changes(self):
        """Apply the current modifications to the image, allowing further edits."""
        if self.image_processor and self.processed_image is not None:
            self.previous_image = self.image_processor.image.copy()
            self.image_processor.image = self.processed_image.copy()
            self.display_image(self.image_processor.image, self.original_label)

            self.undo_action.setEnabled(True)
            self.next_image = None 
            self.redo_action.setEnabled(False)

    
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

    def save_image(self, format):
        """Save the processed image in the selected format."""
        if self.processed_image is None:
            QMessageBox.warning(self, "No Image", "Please process an image before saving.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, f"Save Image as {format.upper()}", "", f"Images (*.{format})")
        if file_path:
            if not file_path.endswith(f".{format}"):
                file_path += f".{format}"  # Ensure the file has the correct extension
            
            cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ImageProcessorGUI()
    window.show()
    sys.exit(app.exec())