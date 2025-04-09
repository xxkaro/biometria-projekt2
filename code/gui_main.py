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
        self.binary_kernel = None

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

        # Sliders layout
        self.slider_layout = QVBoxLayout()

        # Binarization slider
        self.binarization_label = QLabel("Binarization Threshold:")
        self.binarization_slider = QSlider(Qt.Orientation.Horizontal)
        self.binarization_slider.setMinimum(0)
        self.binarization_slider.setMaximum(255)
        self.binarization_slider.setValue(128) 
        self.binarization_slider.valueChanged.connect(self.binarization)

        # Add sliders to the layout
        self.slider_layout.addWidget(self.binarization_label)
        self.slider_layout.addWidget(self.binarization_slider)
        self.slider_layout.setSpacing(4)

        # Buttons layout
        self.button_layout = QVBoxLayout()


        # Morphological Operations
        self.morphological_operations_layout = QHBoxLayout()
        #add dropdown to choose kernel shape
        self.kernel_shape_label = QLabel("Select Kernel Shape:")
        self.kernel_shape_combo = QComboBox()
        self.kernel_shape_combo.addItems(["square", "cross", "vertical_line", "horizontal_line"])
        self.morphological_operations_layout.addWidget(self.kernel_shape_label)
        self.morphological_operations_layout.addWidget(self.kernel_shape_combo)
        self.binary_kernel = self.kernel_shape_combo.currentText()


        self.erosion_button = QPushButton("Errosion")
        self.erosion_button.clicked.connect(self.erosion)
        self.erosion_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.morphological_operations_layout.addWidget(self.erosion_button)

        self.dilation_button = QPushButton("Dilation")
        self.dilation_button.clicked.connect(self.dilation)
        self.dilation_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.morphological_operations_layout.addWidget(self.dilation_button)

        self.opening_button = QPushButton("Opening")
        self.opening_button.clicked.connect(self.opening)
        self.opening_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.morphological_operations_layout.addWidget(self.opening_button)

        self.closing_button = QPushButton("Closing")
        self.closing_button.clicked.connect(self.closing)
        self.closing_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.morphological_operations_layout.addWidget(self.closing_button)

        
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
        self.main_layout.addLayout(self.slider_layout)
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

    
    def binarization(self):
        """Apply binarization based on slider value."""
        if self.image_processor:
            threshold_value = self.binarization_slider.value()
            self.processed_image = self.image_processor.binarization(threshold_value)
            self.display_image(self.processed_image, self.processed_label)

    def erosion(self):
        """Apply erosion to the image."""
        if self.image_processor:
            self.processed_image = self.image_processor.convolute_binary(self.image_processor.image, self.binary_kernel,'erosion')
            self.display_image(self.processed_image, self.processed_label)
    
    def dilation(self):
        """Apply dilation to the image."""
        if self.image_processor:
            self.processed_image = self.image_processor.convolute_binary(self.image_processor.image, self.binary_kernel,'dilation')
            self.display_image(self.processed_image, self.processed_label)

    def opening(self):
        """Apply opening to the image."""
        if self.image_processor:
            self.processed_image = self.image_processor.opening(self.binary_kernel)
            self.display_image(self.processed_image, self.processed_label)

    def closing(self):
        """Apply closing to the image."""     
        if self.image_processor:
            self.processed_image = self.image_processor.closing(self.binary_kernel)
            self.display_image(self.processed_image, self.processed_label)


    
    def apply_changes(self):
        """Apply the current modifications to the image, allowing further edits."""
        if self.image_processor and self.processed_image is not None:
            self.previous_image = self.image_processor.image.copy()
            self.image_processor.image = self.processed_image.copy()
            self.display_image(self.image_processor.image, self.original_label)

            self.undo_action.setEnabled(True)
            self.next_image = None 
            self.redo_action.setEnabled(False)

            self.reset_slider_values()
    
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
            self.reset_slider_values()

    def reset_slider_values(self):
        """Reset the slider values."""
        self.binarization_slider.setValue(128)

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