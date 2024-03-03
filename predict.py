from keras.models import load_model

import matplotlib.pyplot as plt

piCamera = False
if picamera == True:
    from picamera import PiCamera
    import time
    from picamera.array import PiRGBArray

import argparse
import script.utils

def parse_args():
    desc = "Vehicle Classification"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='model/vehicle_model_saved.h5', help='Where Is Model File?')
    parser.add_argument('--img', type=str, default='data/1.jpg', help='What Is Images Path?')

    return parser.parse_args()

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
                camera = PiCamera()
                rawImage = PiRGBArray(camera)
                time.sleep(0.1)
                camera.capture(rawImage, format = "rgb")
                image = rawImage.array
            else:
                image = script.utils.load_image(args.img)
            # Predict Image Based On Model
            label = model.predict(image)
            # Print Result
            print("Predicted Class (0 - Wide Truck , x- Others): ", round(label[0][0], 2))
            
            if round(label[0][0], 2) == 0:
                # displaying the image 
                plt.imshow(image)
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
