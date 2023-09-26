"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from qtpy import QtWidgets
from qtpy import QtCore
from ufish.api import UFish
import napari
from concurrent.futures import ThreadPoolExecutor


class InferenceWidget(QtWidgets.QWidget):
    predict_done_signal = QtCore.Signal(object)

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
        self.run_btn = btn

        select_line = QtWidgets.QHBoxLayout()
        select_line.addWidget(QtWidgets.QLabel("Select input:"))
        self.layer_select = QtWidgets.QComboBox()
        self._update_layer_select()
        self.layer_select.activated.connect(self._update_layer_select)
        select_line.addWidget(self.layer_select)

        input_axes_line = QtWidgets.QHBoxLayout()
        input_axes_line.addWidget(QtWidgets.QLabel("Input axes(optional):"))
        self.input_axes = QtWidgets.QLineEdit("")
        input_axes_line.addWidget(self.input_axes)

        blend_3d_line = QtWidgets.QHBoxLayout()
        blend_3d_line.addWidget(QtWidgets.QLabel("Blend 3D:"))
        self.blend_3d_checkbox = QtWidgets.QCheckBox()
        blend_3d_line.addWidget(self.blend_3d_checkbox)

        batch_size_line = QtWidgets.QHBoxLayout()
        batch_size_line.addWidget(QtWidgets.QLabel("Batch size:"))
        self.batch_size_box = QtWidgets.QSpinBox()
        self.batch_size_box.setValue(4)
        batch_size_line.addWidget(self.batch_size_box)

        p_thresh_line = QtWidgets.QHBoxLayout()
        p_thresh_line.addWidget(QtWidgets.QLabel("p threshold:"))
        self.p_thresh_box = QtWidgets.QDoubleSpinBox()
        self.p_thresh_box.setValue(0.5)
        p_thresh_line.addWidget(self.p_thresh_box)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        layout.addLayout(select_line)
        layout.addLayout(input_axes_line)
        layout.addLayout(blend_3d_line)
        layout.addLayout(batch_size_line)
        layout.addLayout(p_thresh_line)
        layout.addWidget(btn)

        self.ufish = UFish()
        self.ufish.load_weights()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.predict_done_signal.connect(self._on_predict_done)

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
            input_axes = self.input_axes.text() or None
            blend_3d = self.blend_3d_checkbox.isChecked()
            batch_size = self.batch_size_box.value()
            p_thresh = self.p_thresh_box.value()
            self.run_predict(
                layer.data,
                axes=input_axes,
                blend_3d=blend_3d,
                batch_size=batch_size,
                intensity_threshold=p_thresh,
            )

    def run_predict(self, *args, **kwargs):
        self.run_btn.setText("Running...")
        self.run_btn.setEnabled(False)

        def run():
            spots, enh_img = self.ufish.predict(*args, **kwargs)
            self.predict_done_signal.emit((spots, enh_img))
        self.executor.submit(run)

    def _on_predict_done(self, res):
        spots, enh_img = res
        self.viewer.add_image(enh_img, name="enhanced")
        self.viewer.add_points(
            spots, name="spots",
            face_color="blue",
            size=5, opacity=0.5)
        self.run_btn.setText("Run")
        self.run_btn.setEnabled(True)

    def _update_layer_select(self):
        self.layer_select.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.layer_select.addItem(layer.name)
        self.layer_select.update()
        n = self.layer_select.count()
        if n > 0:
            self.layer_select.setCurrentIndex(n-1)
