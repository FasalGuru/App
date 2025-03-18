import torch
import torch.nn as nn
import sys
import os
import pickle

HIDDEN_NEURONS = 10


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input_layer = nn.Linear(7, HIDDEN_NEURONS)
        self.hidden_layer = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.output = nn.Linear(HIDDEN_NEURONS, 22)

        self.relu = nn.ReLU()  # ReLU for hidden layers
        self.softmax = nn.Softmax(dim=1)  # Softmax for multi-class classification

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer(x))
        x = self.output(x)
        x = self.softmax(x)

        return x

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)


class Detector:
    Tabular = Model()
    def __init__(self):
        Detector.Tabular.load_state_dict(torch.load(resource_path("models/tabular_sowing_classification.pt")))

    def predict_class(self, prediction):
        with open(resource_path("models/label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        return label_encoder.inverse_transform([prediction])[0]

