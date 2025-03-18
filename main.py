from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QIcon
import sys
import os

sys.path.append(os.path.abspath("./widgets"))
from toolbar import ToolBar
from tab_widget import TabWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fasal Guru")
        self.setWindowIcon(QIcon("./assets/Fasal.png"))

        self.addToolBar(ToolBar(self))
        self.setCentralWidget(TabWidget())

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


