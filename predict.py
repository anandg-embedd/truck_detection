from keras.models import load_model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

picamera = False
if picamera == True:
    #from picamera import PiCamera
    import time
    #from picamera.array import PiRGBArray
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
    parser.add_argument('--img', type=str, default='dataset/test/Audi/909.jpg', help='What Is Images Path?')
    #parser.add_argument('--img', type=str, default='dataset/test/Toyota Innova/99.jpg', help='What Is Images Path?')

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
            
            # Load Model
            model = load_model(args.model)
            # Convert Image To Numpy Array
            if picamera == True:
##                camera = PiCamera()
##                rawImage = PiRGBArray(camera)
##                time.sleep(0.1)
##                camera.capture(rawImage, format = "rgb")
##                image = rawImage.array
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
            # Predict Image Based On Model
            label = model.predict(image)
            # Print Result
            print("Predicted Class (1 - Wide Truck , x- Others): ", round(label[0][0], 2))
            
            if round(label[0][0], 2) == 1:
                # displaying the image 
                plt.imshow(disp_img)
                plt.title('Wide truck detected',  
                                     fontweight ="bold")
                plt.show()
                print("Wide Truck detected")
            if picamera == False:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
