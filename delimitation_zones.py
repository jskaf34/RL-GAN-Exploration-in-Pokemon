import sys
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QLineEdit, QInputDialog
from PyQt5.QtGui import QPixmap, QPainter, QColor
from PyQt5.QtCore import Qt, QRect


class ImageLabel(QLabel):
    def __init__(self, image_path, max_width, max_height):
        super().__init__()
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio)
        self.setPixmap(pixmap)
        self.rectangles = []
        self.zone_names = []
        self.start_point = None
        self.end_point = None

    def mousePressEvent(self, event):
        self.start_point = event.pos()

    def mouseReleaseEvent(self, event):
        self.end_point = event.pos()
        rect = QRect(self.start_point, self.end_point)
        self.rectangles.append(rect)
        self.start_point = None
        self.end_point = None
        self.get_zone_name()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(Qt.red)
        for rect in self.rectangles:
            painter.drawRect(rect)

    def get_zone_name(self):
        zone_name, ok = QInputDialog.getText(self, 'Zone Name', 'Enter a name for the zone:')
        if ok:
            self.zone_names.append(zone_name)


class MainWindow(QMainWindow):
    def __init__(self, image_path, max_width, max_height):
        super().__init__()
        self.setWindowTitle("Zone Selector")
        self.setGeometry(100, 100, max_width, max_height)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.image_label = ImageLabel(image_path, max_width, max_height)
        self.layout.addWidget(self.image_label)

        self.show_zones_button = QPushButton("Show Zones")
        self.layout.addWidget(self.show_zones_button)
        self.show_zones_button.clicked.connect(self.show_zones)

        self.save_zones_button = QPushButton("Save Zones")
        self.layout.addWidget(self.save_zones_button)
        self.save_zones_button.clicked.connect(self.save_zones)

        self.zones_history = []

    def show_zones(self):
        print("Zones Dictionary:")
        for idx, (zone, name) in enumerate(zip(self.image_label.rectangles, self.image_label.zone_names)):
            print(f"{name}: {zone.x()}, {zone.y()}, {zone.width()}, {zone.height()}")

    def save_zones(self):
        zones_dict = {}
        for idx, (zone, name) in enumerate(zip(self.image_label.rectangles, self.image_label.zone_names)):
            zones_dict[name] = {
                "x": zone.x(),
                "y": zone.y(),
                "width": zone.width(),
                "height": zone.height()
            }
        self.zones_history.append(zones_dict)
        with open("zones.json", "w") as f:
            json.dump(zones_dict, f)
        print("Zones saved as 'zones.json'.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    image_path = "pokemap_full_bw.png"  
    max_width = 800  # Set maximum width for the displayed image
    max_height = 800  # Set maximum height for the displayed image
    window = MainWindow(image_path, max_width, max_height)
    window.show()
    sys.exit(app.exec_())
