"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import typing as T
from qtpy import QtWidgets
from qtpy import QtCore
from ufish.api import UFish
import napari
from concurrent.futures import ThreadPoolExecutor

huggingface_repo = "https://huggingface.co/GangCaoLab/U-FISH"
github_repo = "https://github.com/UFISH-Team/U-FISH/"
napari_plugin_repo = "https://github.com/UFISH-Team/napari-ufish"


inference_help_text = f"""\
This is the inference widget for U-FISH napari plugin.
U-FISH is a deep learning based tool for detecting spots in FISH images.

Parameters:
    weight_file: The weight file for the model.
        If None, will use the default weight file.
    input_axes: The axes of the image.
        For example, 'czxy' for a 4D image,
        'yx' for a 2D image.
        If None, will try to infer the axes from the shape.
    blend_3d: Whether to blend the 3D image.
        Used only when the image contains a z axis.
        If True, will blend the 3D enhanced images along
        the z, y, x axes.
    p_thresh: The threshold for the probability map.
        Range from 0 to 1, higher value will result in
        less spots.
    batch_size: The batch size for inference.
        Used only when the image dimension is 3 or higher.
    chunking: Whether to use chunking for processing.
        If True, will infer the image chunk by chunk.
    chunk_size: The chunk size for processing.
        For example, (1, 512, 512) for a 3D image,
        (512, 512) for a 2D image.
        Using 'image' as a dimension will use the whole image
        as a chunk. For example, (1, 'image', 'image') for a 3D image,
        ('image', 'image', 'image', 512, 512) for a 5D image.
        If None, will use the default chunk size.

You can download the pretrained model weights from:
    {huggingface_repo}
For more information, please visit:
    {github_repo}
    {napari_plugin_repo}
"""


train_widget_help_text = f"""\
This is the train widget for U-FISH napari plugin.
U-FISH is a deep learning based tool for detecting spots in FISH images.

Parameters:
    weight_file(required): The weight file for the model.
        The PyTorch weight file(.pth) is required.
        You can download the pretrained model weights from:
            {huggingface_repo}
    train_dataset(required): The directory for the training dataset.
        The directory should contain the images and the corresponding
        label csv files.
    valid_dataset(required): The directory for the validation dataset.
        The directory should contain the images and the corresponding
        label csv files.
    model_save_dir(required): The directory for saving the trained model.
    data_augmentation: Whether to use data augmentation for training.
    num_epochs: The number of epochs for training.
    batch_size: The batch size for training.
    learning_rate: The learning rate for training.

You can download the pretrained model weights from:
    {huggingface_repo}
For more information, please visit:
    {github_repo}
    {napari_plugin_repo}
"""


class HelpDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, text=""):
        super().__init__(parent)

        # Set the dialog title
        self.setWindowTitle("Help")

        # Create a text edit widget with the help text
        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setPlainText(text)
        self.text_edit.setReadOnly(True)

        # Create a button box with an OK button
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)

        # Add the text edit and button box to the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(button_box)
        self.setLayout(layout)
        self.setFixedWidth(600)


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
        self._init_layout()
        self.ufish = UFish()
        self.ufish.load_weights()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.predict_done_signal.connect(self._on_predict_done)

    def _init_layout(self):
        btn = QtWidgets.QPushButton("Run")
        btn.clicked.connect(self._on_run_click)
        self.run_btn = btn

        select_line = QtWidgets.QHBoxLayout()
        select_line.addWidget(QtWidgets.QLabel("Select input:"))
        self.layer_select = QtWidgets.QComboBox()
        self._update_layer_select()
        self.layer_select.activated.connect(self._update_layer_select)
        select_line.addWidget(self.layer_select)

        weight_file_line = QtWidgets.QHBoxLayout()
        weight_file_line.addWidget(QtWidgets.QLabel("Weight file(optional):"))
        self.weight_file_button = QtWidgets.QPushButton("Open")
        self.weight_file_button.clicked.connect(self._on_weight_file_click)
        weight_file_line.addWidget(self.weight_file_button)

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
        self.p_thresh_box.setSingleStep(0.1)
        self.p_thresh_box.setRange(0.0, 1.0)
        p_thresh_line.addWidget(self.p_thresh_box)

        chunking_line = QtWidgets.QHBoxLayout()
        chunking_line.addWidget(QtWidgets.QLabel("Chunking:"))
        self.chunking_checkbox = QtWidgets.QCheckBox()
        chunking_line.addWidget(self.chunking_checkbox)

        chnksize_line = QtWidgets.QHBoxLayout()
        chnksize_line.addWidget(QtWidgets.QLabel("Chunk size(optional):"))
        self.chunk_size = QtWidgets.QLineEdit("")
        chnksize_line.addWidget(self.chunk_size)

        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self._show_help_dialog)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        layout.addLayout(select_line)
        layout.addLayout(weight_file_line)
        layout.addLayout(input_axes_line)
        layout.addLayout(blend_3d_line)
        layout.addLayout(batch_size_line)
        layout.addLayout(p_thresh_line)
        layout.addLayout(chunking_line)
        layout.addLayout(chnksize_line)
        layout.addWidget(self.help_button)
        layout.addWidget(btn)

    def _show_help_dialog(self):
        dialog = HelpDialog(self, text=inference_help_text)
        dialog.exec_()

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
            chunking = self.chunking_checkbox.isChecked()
            chunk_size = self.chunk_size.text() or None
            if isinstance(chunk_size, str):
                chunk_size = eval(chunk_size)
            self.run_predict(
                layer.name,
                layer.data,
                chunking=chunking,
                chunk_size=chunk_size,
                axes=input_axes,
                blend_3d=blend_3d,
                batch_size=batch_size,
                intensity_threshold=p_thresh,
            )

    def _on_weight_file_click(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open weight file", "",
            "All Files (*);;ONNX Files (*.onnx);;PyTorch Files (*.pth)",
            options=options)
        if file_name:
            print(file_name)
            self.ufish.load_weights(file_name)

    def run_predict(
            self, name, data,
            chunking: bool = False,
            chunk_size: T.Optional[tuple] = None,
            *args, **kwargs):
        self._toggle_run_btn(False)

        is_mul_scale = 'MultiScaleData' in str(type(data))
        if is_mul_scale:
            data = data[0]

        def run():
            try:
                if not chunking:
                    spots, enh_img = self.ufish.predict(
                        data, *args, **kwargs)
                else:
                    kwargs["chunk_size"] = chunk_size
                    spots, enh_img = self.ufish.predict_chunks(
                        data, *args, **kwargs)
            except Exception as e:
                self.predict_done_signal.emit((e, None, None))
            self.predict_done_signal.emit((name, spots, enh_img))
        self.executor.submit(run)

    def _on_predict_done(self, res):
        name, spots, enh_img = res
        if isinstance(name, Exception):
            self._toggle_run_btn(True)
            raise name
        self.viewer.add_image(
            enh_img, name=f"{name}.enhanced")
        self.viewer.add_points(
            spots, name=f"{name}.spots",
            face_color="blue",
            size=5, opacity=0.5)
        self._toggle_run_btn(True)

    def _toggle_run_btn(self, enabled):
        if enabled:
            self.run_btn.setText("Run")
            self.run_btn.setEnabled(True)
        else:
            self.run_btn.setText("Running...")
            self.run_btn.setEnabled(False)

    def _update_layer_select(self):
        self.layer_select.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.layer_select.addItem(layer.name)
        self.layer_select.update()
        n = self.layer_select.count()
        if n > 0:
            self.layer_select.setCurrentIndex(n-1)


class TrainWidget(QtWidgets.QWidget):
    done_signal = QtCore.Signal(object)

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer
        self.train_dataset_path = None
        self.valid_dataset_path = None
        self.weight_loaded = False
        self.model_save_dir = None
        self._init_layout()
        self.ufish = UFish()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.done_signal.connect(self._on_train_done)

    def _init_layout(self):
        layout = QtWidgets.QVBoxLayout()
        torch_vesion_label = self._check_torch_version()
        layout.addWidget(torch_vesion_label)
        layout.addSpacing(15)

        self.weight_file_button = QtWidgets.QPushButton(
            "Open weight file")
        self.weight_file_button.clicked.connect(self._on_weight_file_click)
        self.weight_file_label = QtWidgets.QLabel("None")
        layout.addWidget(QtWidgets.QLabel("Weight file(required):"))
        layout.addWidget(self.weight_file_button)
        layout.addWidget(self.weight_file_label)
        layout.addSpacing(10)

        self.train_dataset_button = QtWidgets.QPushButton(
            "Open train directory")
        self.train_dataset_button.clicked.connect(self._on_open_train_dataset)
        self.train_dataset_label = QtWidgets.QLabel("None")
        layout.addWidget(QtWidgets.QLabel("Train dataset(required):"))
        layout.addWidget(self.train_dataset_button)
        layout.addWidget(self.train_dataset_label)
        layout.addSpacing(10)

        self.valid_dataset_button = QtWidgets.QPushButton(
            "Open validation directory")
        self.valid_dataset_button.clicked.connect(
            self._on_open_validation_dataset)
        self.valid_dataset_label = QtWidgets.QLabel("None")
        layout.addWidget(QtWidgets.QLabel("Validation dataset(required):"))
        layout.addWidget(self.valid_dataset_button)
        layout.addWidget(self.valid_dataset_label)
        layout.addSpacing(10)

        self.model_save_dir_button = QtWidgets.QPushButton(
            "Open model save directory")
        self.model_save_dir_button.clicked.connect(
            self._on_open_model_save_dir)
        model_save_dir_line = QtWidgets.QHBoxLayout()
        model_save_dir_line.addWidget(
            QtWidgets.QLabel("Open model save directory"))
        self.model_save_dir_label = QtWidgets.QLabel("None")
        layout.addWidget(QtWidgets.QLabel("Model save directory(required):"))
        layout.addWidget(self.model_save_dir_button)
        layout.addWidget(self.model_save_dir_label)
        layout.addSpacing(15)

        data_argu_line = QtWidgets.QHBoxLayout()
        data_argu_line.addWidget(QtWidgets.QLabel("Data augmentation:"))
        self.data_argu_checkbox = QtWidgets.QCheckBox()
        data_argu_line.addWidget(self.data_argu_checkbox)
        layout.addLayout(data_argu_line)

        num_epochs_line = QtWidgets.QHBoxLayout()
        num_epochs_line.addWidget(QtWidgets.QLabel("Number of epochs:"))
        self.num_epochs_box = QtWidgets.QSpinBox()
        self.num_epochs_box.setValue(50)
        num_epochs_line.addWidget(self.num_epochs_box)
        layout.addLayout(num_epochs_line)

        batch_size_line = QtWidgets.QHBoxLayout()
        batch_size_line.addWidget(QtWidgets.QLabel("Batch size:"))
        self.batch_size_box = QtWidgets.QSpinBox()
        self.batch_size_box.setValue(8)
        batch_size_line.addWidget(self.batch_size_box)
        layout.addLayout(batch_size_line)

        learning_rate_line = QtWidgets.QHBoxLayout()
        learning_rate_line.addWidget(QtWidgets.QLabel("Learning rate:"))
        self.learning_rate_box = QtWidgets.QDoubleSpinBox()
        self.learning_rate_box.setDecimals(8)
        self.learning_rate_box.setValue(0.0001)
        self.learning_rate_box.setSingleStep(0.0001)
        learning_rate_line.addWidget(self.learning_rate_box)
        layout.addLayout(learning_rate_line)
        layout.addSpacing(15)

        self.train_button = QtWidgets.QPushButton("Train")
        self.train_button.setEnabled(False)
        self.train_button.clicked.connect(self._on_train_click)
        layout.addWidget(self.train_button)
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setEnabled(False)
        layout.addWidget(self.run_button)
        # TODO: implement run
        self.convert_button = QtWidgets.QPushButton("Convert to ONNX")
        self.convert_button.setEnabled(False)
        # TODO: implement convert to onnx
        layout.addWidget(self.convert_button)
        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(
            self._on_help_button_click)
        layout.addWidget(self.help_button)

        self.setLayout(layout)

    def _check_torch_version(self):
        import torch
        version = torch.__version__
        label = QtWidgets.QLabel(f"PyTorch version: {version}")
        if not torch.cuda.is_available():
            label.setText(
                f"PyTorch version: {version}\n"
                "Training requires PyTorch with CUDA support.\n"
                "Make sure GPU is available and CUDA is installed."
            )
            label.setStyleSheet("color: red")
            self.weight_file_button.setEnabled(False)
        return label

    def _on_help_button_click(self):
        dialog = HelpDialog(self, text=train_widget_help_text)
        dialog.exec_()

    def _on_weight_file_click(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open weight file", "",
            "PyTorch Files (*.pth)",
            options=options)
        if file_name:
            print(file_name)
            assert file_name.endswith(".pth"),\
                "Only support PyTorch weights(.pth)"
            self.ufish.load_weights(file_name)
            self.weight_loaded = True
            self.weight_file_label.setText(file_name)
            self.run_button.setEnabled(True)
            self.convert_button.setEnabled(True)
            if self._is_trainable:
                self.train_button.setEnabled(True)

    def _on_open_train_dataset(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open train directory", options=options)
        if dir_name:
            print(dir_name)
            self.train_dataset_label.setText(dir_name)
            self.train_dataset_path = dir_name
            if self._is_trainable:
                self.train_button.setEnabled(True)

    def _on_open_validation_dataset(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open validation directory", options=options)
        if dir_name:
            print(dir_name)
            self.valid_dataset_label.setText(dir_name)
            self.valid_dataset_path = dir_name
            if self._is_trainable:
                self.train_button.setEnabled(True)

    def _on_open_model_save_dir(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open model save directory", options=options)
        if dir_name:
            print(dir_name)
            self.model_save_dir_label.setText(dir_name)
            self.model_save_dir = dir_name

    def _is_trainable(self):
        return self.train_dataset_path is not None and\
            self.valid_dataset_path is not None and\
            self.model_save_dir is not None and\
            self.weight_loaded

    def _on_train_click(self):
        self.train_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.convert_button.setEnabled(False)

        def train():
            try:
                self.ufish.train(
                    self.train_dataset_path,
                    self.valid_dataset_path,
                    model_save_dir=self.model_save_dir,
                    data_augmentation=self.data_argu_checkbox.isChecked(),
                    num_epochs=self.num_epochs_box.value(),
                    batch_size=self.batch_size_box.value(),
                    learning_rate=self.learning_rate_box.value(),
                )
            except Exception as e:
                self.done_signal.emit(e)
            self.done_signal.emit(None)
        self.executor.submit(train)

    def _on_train_done(self, res):
        if isinstance(res, Exception):
            raise res
        self.train_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.convert_button.setEnabled(True)
