

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

tensor_lite_en = True
picamera = 1
webcamera = 2
nocamera = 0
camerain = webcamera
if camerain == picamera:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
elif camerain == webcamera:
    import cv2
    webcam = cv2.VideoCapture(0)
    time.sleep(2)

import argparse
import script.utils
import numpy as np
from skimage import transform

img_width, img_height = 224, 224

def parse_args():
    desc = "Vehicle Classification"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='model/categorical_vehicle_model_saved.h5', help='Where Is Model File?')
    parser.add_argument('--img', type=str, default='dataset/test/others/42.jpg', help='What Is Images Path?')
    #parser.add_argument('--img', type=str, default='dataset/test/Trucks/5.jpg', help='What Is Images Path?')

    return parser.parse_args()


def img_pre_process(np_image):
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (img_width, img_height, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def main():
    args = parse_args()
    if args is None:
        exit()
    try:
        while True:
            

            
            if camerain == picamera:
                # Convert Image To Numpy Array
                camera = PiCamera()
                rawImage = PiRGBArray(camera)
                time.sleep(0.1)
                camera.capture(rawImage, format = "rgb")
                disp_img = rawImage.array
                image = img_pre_process(disp_img)
            elif camerain == webcamera:
                try:
                    check, frame = webcam.read()
                    disp_img=frame;
                    image = img_pre_process(frame)
                except(KeyboardInterrupt):
                    print("Turning off camera.")
                    webcam.release()
                    print("Camera off.")
                    print("Program ended.")
                    cv2.destroyAllWindows()
                    break
            else:
                image = script.utils.load_image(args.img)
                disp_img=mpimg.imread(args.img)

            if tensor_lite_en == True:
                import tensorflow as tf

                # Load the TFLite model and allocate tensors.
                interpreter = tf.lite.Interpreter(model_path="model.tflite")
                interpreter.allocate_tensors()

                # Get input and output tensors.
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Test the model on random input data.
                # input_shape = input_details[0]['shape']
                input_data = image # np.array(np.random.random_sample(input_shape), dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)

                interpreter.invoke()

                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                label = interpreter.get_tensor(output_details[0]['index'])
                print(label)
            else:
                from keras.models import load_model
                # Load Model
                model = load_model(args.model)
                # Predict Image Based On Model
                label = model.predict(image)
                
            # Print Result
            print("Predicted Class (0 to 0.5(1) - Wide Truck , x- Others): ", round(label[0][0], 2))
            
            if round(label[0][0], 2) >= 0 and round(label[0][0], 2) <= 0.5 :
                # displaying the image 
                plt.imshow(disp_img)
                plt.title('Wide truck detected',  
                                     fontweight ="bold")
                plt.show()
                print("Wide Truck detected")
            if camerain == nocamera:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
