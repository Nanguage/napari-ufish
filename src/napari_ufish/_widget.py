"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from qtpy import QtWidgets
from ufish.api import UFish
import napari


class InferenceWidget(QtWidgets.QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer
        self.viewer.layers.events.inserted.connect(
            lambda e: self._update_layer_select())
        self.viewer.layers.events.removed.connect(
            lambda e: self._update_layer_select())
        self.viewer.layers.events.moved.connect(
            lambda e: self._update_layer_select())

        btn = QtWidgets.QPushButton("Run")
        btn.clicked.connect(self._on_run_click)

        select_line = QtWidgets.QHBoxLayout()
        select_line.addWidget(QtWidgets.QLabel("Select input:"))
        self.layer_select = QtWidgets.QComboBox()
        self._update_layer_select()
        self.layer_select.activated.connect(self._update_layer_select)
        select_line.addWidget(self.layer_select)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        layout.addLayout(select_line)
        layout.addWidget(btn)

        self.ufish = UFish()
        self.ufish.load_weights()

    def _on_run_click(self):
        image_layers = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]
        if len(image_layers) > 0:
            idx = self.layer_select.currentIndex()
            layer = image_layers[idx]
            print("Run inference on", layer.name)
            spots, enh_img = self.ufish.predict(layer.data)

            self.viewer.add_image(
                enh_img, name=f"{layer.name}.enhanced")
            self.viewer.add_points(
                spots, name=f"{layer.name}.spots",
                size=5, opacity=0.5)

    def _update_layer_select(self):
        print("update layer select")
        self.layer_select.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.layer_select.addItem(layer.name)
        self.layer_select.update()
        n = self.layer_select.count()
        if n > 0:
            self.layer_select.setCurrentIndex(n-1)
