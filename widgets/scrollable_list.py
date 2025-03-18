from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QListWidget
class ScrollableList(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("QListWidget Example")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout(self)

        # Create a QListWidget (automatically scrollable)
        list_widget = QListWidget(self)

        # Add multiple items to the list
        for i in range(1, 21):
            list_widget.addItem(f"Item {i}")

        layout.addWidget(list_widget)