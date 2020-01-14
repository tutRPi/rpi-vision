# Python
import logging
import os.path
import shutil
import zipfile
# lib
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import PIL
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions


logger = logging.getLogger(__name__)


def load_labels(label_file_path):
    label_dict = {}
    with open(label_file_path) as label_f:
        for line in label_f:
            parts = line.rstrip('\n').split(' ', 1)
            label_dict[int(parts[0])] = parts[1]

    return label_dict


class TeachableMachine():
    def __init__(self, zip_file_path):
        extract_dir = os.path.splitext(zip_file_path)[0]
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        zf = zipfile.ZipFile(zip_file_path)
        zf.extractall(extract_dir)

        labels_file = os.path.join(extract_dir, 'labels.txt')
        self.labels = load_labels(labels_file)
        logging.info('loaded labels: %r', self.labels)

        ## Disable scientific notation for clarity
        # np.set_printoptions(suppress=True)
        saved_model_path = os.path.join(extract_dir, 'model.savedmodel')
        self.model = models.load_model(saved_model_path).signatures['serving_default']

#        logger.info(self.model.summary())
        self.tflite_interpreter = None

    def predict(self, frame):
        # resize to 224x224 as required by TeachableMachine
        im = PIL.Image.fromarray(frame)
        im.resize((224, 224))
        # expand 3D RGB frame into 4D batch
        sample = np.expand_dims(np.asarray(im), axis=0)
        # image = image.resize((224, 224))
        processed_sample = preprocess_input(sample.astype(np.float32))
        features = self.model(tf.constant(processed_sample))

        all_features = list(sorted(features.keys()))
        if len(all_features) == 1:
            feature_key = all_features[0]
        else:
            all_features_str = ', '.join(all_features)
            logging.error('Expected to find 1 feature; actually found:\n%s',
                          all_features_str)
            raise Exception('Want 1 feature, found %d: %s' % (
                len(all_features), all_features_str))

        seq = features[feature_key].numpy()[0]

        decoded = []
        for i, f in enumerate(seq):
            decoded.append([i, self.labels[i], f])
        return [decoded]

    def tflite_convert_from_keras_model_file(self, output_dir='includes/', output_filename='mobilenet_v2_imagenet.tflite', keras_model_file='includes/mobilenet_v2_imagenet.h5'):
        # @todo TFLiteConverter.from_keras_model() is only available in the tf-nightly-2.0-preview build right now
        # https://groups.google.com/a/tensorflow.org/forum/#!searchin/developers/from_keras_model%7Csort:date/developers/Mx_EaHM1X2c/rx8Tm-24DQAJ
        # converter = tf.lite.TFLiteConverter.from_keras_model(self.model_base)
        converter = tf.lite.TFLiteConverter.from_keras_model_file(
            keras_model_file)
        tflite_model = converter.convert()
        if output_dir and output_filename:
            with open(output_dir + output_filename, 'wb') as f:
                f.write(tflite_model)
                logger.info('Wrote {}'.format(output_dir + output_filename))
        return tflite_model

    def tflite_convert_from_keras_model(self, output_dir='includes/', output_filename='mobilenet_v2_imagenet.tflite'):
        # @todo TFLiteConverter.from_keras_model() is only available in the tf-nightly-2.0-preview build right now
        # https://groups.google.com/a/tensorflow.org/forum/#!searchin/developers/from_keras_model%7Csort:date/developers/Mx_EaHM1X2c/rx8Tm-24DQAJ
        # converter = tf.lite.TFLiteConverter.from_keras_model(self.model_base)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model_base)
        tflite_model = converter.convert()
        if output_dir and output_filename:
            with open(output_dir + output_filename, 'wb') as f:
                f.write(tflite_model)
                logger.info('Wrote {}'.format(output_dir + output_filename))
        return tflite_model

    def init_tflite_interpreter(self, model_path='includes/mobilenet_v2_imagenet.tflite'):
        '''
            https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/lite/Interpreter
            This makes the TensorFlow Lite interpreter accessible in Python.
            It is possible to use this interpreter in a multithreaded Python environment, but you must be sure to call functions of a particular instance from only one thread at a time.
            So if you want to have 4 threads running different inferences simultaneously, create an interpreter for each one as thread-local data.
            Similarly, if you are calling invoke() in one thread on a single interpreter but you want to use tensor() on another thread once it is done
            you must use a synchronization primitive between the threads to ensure invoke has returned before calling tensor().

        '''
        self.tflite_interpreter = tf.lite.Interpreter(
            model_path=model_path)
        self.tflite_interpreter.allocate_tensors()
        logger.info('Initialized tflite Python interpreter \n',
                    self.tflite_interpreter)

        self.tflite_input_details = self.tflite_interpreter.get_input_details()
        logger.info('tflite input details \n', self.tflite_input_details)

        self.tflite_output_details = self.tflite_interpreter.get_output_details()
        logger.info('tflite output details \n',
                    self.tflite_output_details)

        return self.tflite_interpreter

    def tflite_predict(self, frame, input_shape=None):
        if not self.tflite_interpreter:
            self.init_tflite_interpreter()

        dtype = self.tflite_input_details[0].get('dtype')

        # expand 3D RGB frame into 4D batch (of 1 item)
        sample = np.expand_dims(frame, axis=0)
        processed_sample = preprocess_input(sample.astype(dtype))

        self.tflite_interpreter.set_tensor(
            self.tflite_input_details[0]['index'], processed_sample)
        self.tflite_interpreter.invoke()

        features = self.tflite_interpreter.get_tensor(
            self.tflite_output_details[0]['index'])
        decoded_features = decode_predictions(features)

        return decoded_features

    def init_training_model(self, train_dir='data/'):

        if self.include_top is True:
            raise ValueError(
                'FATAL: Cannot re-train a model_base initialized with include_top=True. Init with include_top=False if you want to train additional Dense layers')

        conv_base = self.model_base

        # Freeze the convolutional base to prevent its weights from being re-trained
        # If the base is not frozen, weight updates will be propagated through the network -  this would destroy the learned representation benchmarked in mobilenetv2
        conv_base.trainable = False

        model = tf.keras.models.Sequential()
        model.add(conv_base)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='softmax'))


# # Create the array of the right shape to feed into the keras model
# # The 'length' or number of images you can put into the array is
# # determined by the first position in the shape tuple, in this case 1.
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# # Replace this with the path to your image
# image = Image.open('Path to your image')

# # Make sure to resize all images to 224, 224 otherwise they won't fit in the array
# image = image.resize((224, 224))
# image_array = np.asarray(image)

# # Normalize the image
# normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# # Load the image into the array
# data[0] = normalized_image_array

# # run the inference
# prediction = model.predict(data)
# print(prediction)
