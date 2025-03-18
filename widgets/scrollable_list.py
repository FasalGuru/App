from PySide6.QtWidgets import QWidget, QVBoxLayout, QListWidget, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

class ScrollableList(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Recommended Crops")
        self.setGeometry(100, 100, 400, 300)
        self.setToolTip("List of Crops it can recommend")

        layout = QVBoxLayout(self)

        # Title Label
        title_label = QLabel("🌱 Recommended Crops")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("color: #005CBF; margin-bottom: 5px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Create a QListWidget (automatically scrollable)
        list_widget = QListWidget(self)
        crops = [
            'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
            'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
            'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
            'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'
        ]

        # Add multiple items to the list
        for crop in crops:
            list_widget.addItem(f"{crop.upper()}")

        # Apply styling
        list_widget.setStyleSheet("""
            QListWidget {
                border: 2px solid #ccc;
                border-radius: 10px;
                padding: 5px;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }
            QListWidget::item:hover {
                color: #005CBF;
            }
            QListWidget::item:selected {
                color: white;
                font-weight: bold;
                border-radius: 6px;
            }
        """)

        # Add widgets to layout
        layout.addWidget(title_label)
        layout.addWidget(list_widget)
        self.setLayout(layout)
