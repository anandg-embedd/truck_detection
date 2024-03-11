import tensorflow as tf
from keras.models import load_model
import argparse

tensor_lite_train = True;

def parse_args():
    desc = "Vehicle Classification"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='model/categorical_vehicle_model_saved.h5', help='Where Is Model File?')
    #parser.add_argument('--img', type=str, default='dataset/test/others/42.jpg', help='What Is Images Path?')
    parser.add_argument('--img', type=str, default='dataset/test/Trucks/5.jpg', help='What Is Images Path?')

    return parser.parse_args()
if tensor_lite_train == True:
    args = parse_args()
    if args is None:
        exit()
    # Load Model
    model = load_model(args.model)
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('model.tflite', 'wb') as f:
      f.write(tflite_model)



