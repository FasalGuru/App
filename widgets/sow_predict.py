from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QLabel, QFrame, QSizePolicy, QGridLayout
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
import torch

from widgets.scrollable_list import ScrollableList
from widgets.histogram import Histogram
from utils.detector import Detector

class LineInput(QLineEdit):
    def __init__(self, value_range, default=0):
        super().__init__()
        self.__range = {"min": value_range[0], "MAX": value_range[1]}
        self.setText(f"{default}")
        self.textChanged.connect(lambda: self.__enforce_valid_input(self))

        # Input Styling
        self.setStyleSheet("""
            QLineEdit {
                border: 2px solid #ccc;
                border-radius: 6px;
                padding: 6px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #0078D7;
            }
        """)

    def __enforce_valid_input(self, line_edit):
        text = line_edit.text()
        if text:
            try:
                value = float(text)
                if value < self.__range["min"] or value > self.__range["MAX"]:
                    line_edit.setText(text[:-1])
            except ValueError:
                line_edit.setText(text[:-1])

class SowPredictTab(QWidget):
    def __init__(self):
        super().__init__()
        self.__layout = QGridLayout()
        self.__form_values = {}
        self.detector = Detector()

        # Create form layout
        self.__form = self.__parameters_form()
        self.__result = QLabel()
        self.__histogram = Histogram()

        # Form + Submit Button
        form_layout = QGridLayout()
        form_layout.addLayout(self.__form, 0, 0, 1, 2)  # Span two columns
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.__submit_form)

        # Styling Submit Button
        submit_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                font-size: 16px;
                padding: 8px 16px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #005CBF;
            }
        """)
        form_layout.addWidget(submit_button, 1, 0, 1, 2)  # Submit button below form

        # Result Layout
        result_label = QLabel("Result:")
        result_label.setStyleSheet(
            """
                QLabel{
                    border: 1px solid white;
                    padding: 16px;
                    border-radius: 4px;
                }
            """
        )
        result_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.__result.setFont(QFont("Arial", 16))

        # Add widgets to the grid layout
        self.__layout.addLayout(form_layout, 0, 0)  # Form in first row, first column
        self.__layout.addWidget(ScrollableList(), 0, 1)  # Scrollable list in first row, second column

        self.__layout.addWidget(result_label, 1, 0, Qt.AlignmentFlag.AlignCenter)  # Result label on second row, first column
        self.__layout.addWidget(self.__result, 2, 0, Qt.AlignmentFlag.AlignCenter)  # Result text below the label

        self.__layout.addWidget(self.__histogram, 1, 1, 2, 1)  # Histogram spans two rows in second column

        self.setLayout(self.__layout)

    def __parameters_form(self):
        form_layout = QFormLayout()

        self.nitrogen_input = LineInput((0, 200))
        self.phosphorous_input = LineInput((0, 200))
        self.potassium_input = LineInput((0, 250))
        self.temperature_input = LineInput((-5, 60))
        self.humidity_input = LineInput((0, 100))
        self.ph_input = LineInput((0, 14))
        self.rainfall_input = LineInput((0, 400))

        labels = [
            ("Nitrogen (N) Ratio in the soil:", self.nitrogen_input),
            ("Phosphorous (P) Ratio in the soil:", self.phosphorous_input),
            ("Potassium (K) Ratio in the soil:", self.potassium_input),
            ("Temperature (°C):", self.temperature_input),
            ("Humidity (%):", self.humidity_input),
            ("pH of the soil (0-14):", self.ph_input),
            ("Rainfall (mm):", self.rainfall_input),
        ]

        for text, widget in labels:
            label = QLabel(text)
            label.setFont(QFont("Arial", 12))
            form_layout.addRow(label, widget)

        return form_layout

    def __submit_form(self):
        # Collect and print data
        self.__form_values = {
            "N": float(self.nitrogen_input.text()) / 140,
            "P": float(self.phosphorous_input.text()) / 145,
            "K": float(self.potassium_input.text()) / 205,
            "temperature": float(self.temperature_input.text()) / 43.67549305,
            "humidity": float(self.humidity_input.text()) / 99.98187601,
            "ph": float(self.ph_input.text()) / 9.93509073,
            "rainfall": float(self.rainfall_input.text()) / 298.5601175,
        }

        print("Form Data Submitted:")
        for key, value in self.__form_values.items():
            print(f"{key}: {value}")

        input_tensor = torch.tensor([list(self.__form_values.values())], dtype=torch.float32)
        with torch.no_grad():
            prediction = self.detector.Tabular(input_tensor)

        predicted_class_num = prediction.argmax(dim=1).item()
        result_text = self.detector.predict_class(predicted_class_num)
        print(f"Predicted Class: {result_text}")

        self.__result.setText(result_text.upper())
        self.__histogram.plot_histogram(data=list(self.__form_values.values()))
