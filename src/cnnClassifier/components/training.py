from cnnClassifier.entity.config_entity import TrainingConnfig
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model

class Training:
    def __init__(self, config : TrainingConnfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            str(self.config.updated_base_model_path)
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )


   
    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation= 'bilinear'
        )

        valid_datagenerator= tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator=valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset='validation',
            shuffle=False,
            **dataflow_kwargs            
        )

        if self.config.params_is_agumentation:
            train_datagenerator=tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )

        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset='validation',
            shuffle=True,
            **dataflow_kwargs  
        )

    @staticmethod
    def save_model(path : Path, model : tf.keras.Model):
        model.save(str(path))
        

    def train(self, callback_list: list):
        self.steps_per_epochs =self.train_generator.samples //  self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            steps_per_epoch=self.steps_per_epochs,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
            callbacks=callback_list
        )

        self.save_model(path=self.config.trained_model_path, model =self.model)
