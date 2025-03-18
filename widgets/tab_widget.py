from PySide6.QtWidgets import QTabWidget, QWidget
from widgets.sow_predict import SowPredictTab

class TabWidget(QTabWidget):
    def __init__(self):
        super().__init__()

        # Add tabs
        self.__sow_tab()
        self.__disease_tab()

    def __sow_tab(self):
        self.addTab(SowPredictTab(), "Sowing Prediction")

    def __disease_tab(self):
        disease_tab = QWidget()
        self.addTab(disease_tab, "Disease Prediction")

    def getSelectedTab(self):
        return self.currentWidget()
