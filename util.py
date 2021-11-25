from cv2 import imshow, waitKey
import numpy as np

def show(name, img):
    imshow(name, img)
    waitKey(0)
    
def show_colour(colour, name="Colours"):
    image = np.zeros((300, 300, 3), np.uint8)
    image[:] = colour
    show(name, image)