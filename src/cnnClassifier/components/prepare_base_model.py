import os
from zipfile import ZipFile
import tensorflow as tf
from urllib import request
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        tf.keras.saving.save_model(filepath=self.config.base_model_path, model=self.model)

    @staticmethod
    def prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None and freeze_till > 1:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.get_base_model()
        self.full_model = self.prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(self.config.updated_base_model_path, self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
