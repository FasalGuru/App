from PySide6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QLineEdit, QPushButton
from PySide6.QtGui import QDoubleValidator

# N - ratio of Nitrogen content in soil
# P - ratio of Phosphorous content in soil
# K - ratio of Potassium content in soil
# temperature - temperature in degree Celsius
# humidity - relative humidity in %
# ph - ph value of the soil
# rainfall - rainfall in mm

# Standardizing Values
# 140
# 145
# 205
# 43.67549305
# 99.98187601
# 9.93509073
# 298.5601175

class LineInput(QLineEdit):
    def __enforce_valid_input(self, line_edit):
        text = line_edit.text()
        if text:
            try:
                value = float(text)
                if value < self.__range["min"] or value > self.__range["MAX"]:
                    line_edit.setText(text[:-1])  # Remove last character if out of range
            except ValueError:
                line_edit.setText(text[:-1])  # Remove invalid characters

    def __init__(self, value_range, default = 0):
        super().__init__()
        self.__range = {
            "min": value_range[0],
            "MAX": value_range[1]
        }
        self.setText(f"{default}")
        self.textChanged.connect(lambda: self.__enforce_valid_input(self))


class SowPredictTab(QWidget):
    def __init__(self):
        super().__init__()
        self.__layout = QVBoxLayout()
        self.__form_values = {}

        # Create form layout
        self.__form = self.__parameters_form()
        self.__layout.addLayout(self.__form)

        # Submit Button
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.__submit_form)
        self.__layout.addWidget(submit_button)

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

        form_layout.addRow("Nitrogen (N) Ratio in the soil: ", self.nitrogen_input)
        form_layout.addRow("Phosphorous (P) Ratio in the soil: ", self.phosphorous_input)
        form_layout.addRow("Potassium (K) Ratio in the soil: ", self.potassium_input)
        form_layout.addRow("Temperature (°C): ", self.temperature_input)
        form_layout.addRow("Humidity (%): ", self.humidity_input)
        form_layout.addRow("pH of the soil (0-14): ", self.ph_input)
        form_layout.addRow("Rainfall (mm): ", self.rainfall_input)

        return form_layout

    def __submit_form(self):
        # Collect and print data
        self.__form_values = {
            "N": float(self.nitrogen_input.text())/140,
            "P": float(self.phosphorous_input.text())/145,
            "K": float(self.potassium_input.text())/205,
            "temperature": float(self.temperature_input.text())/43.67549305,
            "humidity": float(self.humidity_input.text())/99.98187601,
            "ph": float(self.ph_input.text())/9.93509073,
            "rainfall": float(self.rainfall_input.text())/298.5601175,
        }

        print("Form Data Submitted:")
        for key, value in self.__form_values.items():
            print(f"{key}: {value}")

