from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPixmap, QPainter
from PySide6.QtCore import Qt

import sys
import os


class DiseasePredictTab(QWidget):
    def get_bg(self, img_path):
        if hasattr(sys, "_MEIPASS"):
            img_path = os.path.join(sys._MEIPASS, img_path)
        else:
            img_path = os.path.abspath(img_path)
        img_path = img_path.replace("\\", "/")

        if not os.path.exists(img_path):
            print("None Exists")
            return None
        return img_path
    
    def __init__(self):
        super().__init__()
        self.__bg_path = self.get_bg("./assets/coming_soon.png")
        print(self.__bg_path)
        self.__bg_img = QPixmap(self.__bg_path)

    
    def paintEvent(self, event):
        painter = QPainter(self)
        scaled_pixmap = self.__bg_img.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)