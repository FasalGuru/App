from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

class Histogram(QWidget):
    def __init__(self):
        super().__init__()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

        self.plot_histogram()  # Initial Plot

    def plot_histogram(self, max_values=[1, 1, 1, 1, 1, 1, 1], data=None):
        self.figure.clear()  # Clear previous figure

        ax = self.figure.add_subplot(1, 1, 1)
        x = np.arange(len(max_values))

        if data is None:  # Single Histogram Case
            ax.bar(x, max_values, color='blue', alpha=0.7)
            ax.set_title("Ideal Values")

        else:  # Two Overlapping Histograms
            ax.bar(x - 0.2, max_values, width=0.4, color='blue', alpha=0.7, label="Ideal")
            ax.bar(x + 0.2, data, width=0.4, color='red', alpha=0.7, label="Actual")
            ax.set_title("Ideal vs Actual Values")
            ax.legend()

        ax.axis('off')  # Hide axes
        self.canvas.draw()  # Update the plot


