from PySide6.QtWidgets import QToolBar, QLineEdit, QFileDialog, QMessageBox
from PySide6.QtGui import QAction, QIcon, QPixmap, QImage, QDesktopServices
from PySide6.QtCore import Qt, QSize
import sys
import os


class ToolBar(QToolBar):
    def __init__(self, main_window):
        super().__init__()

        self.MainWindow = main_window
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.setIconSize(QSize(24, 24))
        self.setMovable(False)

        # Actions
        self.__screenshot()
        self.__version()
        self.__info()

    def __get_icon(self, icon_path):
        if hasattr(sys, "_MEIPASS"):  # Running as an .exe
            icon_path = os.path.join(sys._MEIPASS, icon_path)
        else:  # Running in normal Python
            icon_path = os.path.abspath(icon_path)
        
        icon_path = icon_path.replace("\\", "/")
        if not os.path.exists(icon_path):
            return None
        return QIcon(icon_path)

    def __screenshot_action(self):
        screenshot_bitmap = QPixmap(
            self.MainWindow.centralWidget().getSelectedTab().size()
        )
        self.MainWindow.centralWidget().getSelectedTab().render(screenshot_bitmap)
        save_path, _ = QFileDialog.getSaveFileName(
            self.MainWindow,
            "Save Screenshot",
            "screenshot.png",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
        )
        if save_path:  # Ensure user selected a valid path
            if not screenshot_bitmap.save(save_path):
                QMessageBox.critical(
                    self.MainWindow, "Error", "Failed to save screenshot!"
                )

    def __version_action(self):
        QMessageBox.information(self.MainWindow, "Version Information", 
         "Fasal Guru:\n"
         "This is version 1.0.0 of the Fasal Guru Desktop Application\n"
         "Release Date: 18/03/2025 \n"
         "© 2025 "
        )
    
    def __info_action(self):
        url = "https://pytorch.org/tutorials/beginner/saving_loading_models.html" ## Temporay
        QDesktopServices.openUrl(url)

    def __screenshot(self):
        ss_action = QAction(self.__get_icon("./assets/screenshot.png"), "", self)
        ss_action.triggered.connect(self.__screenshot_action)
        self.addAction(ss_action)

    def __version(self):
        version_action = QAction(self.__get_icon("./assets/version.png"), "", self)
        version_action.triggered.connect(self.__version_action)
        self.addAction(version_action)

    def __info(self):
        info_action = QAction(self.__get_icon("./assets/information.png"), "", self)
        info_action.triggered.connect(self.__info_action)
        self.addAction(info_action)
