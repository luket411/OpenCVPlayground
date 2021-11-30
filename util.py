from cv2 import imshow, waitKey
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(name, img):
    imshow(name, img)
    waitKey(0)
    
def show_colour(colour, name="Colours"):
    image = np.zeros((300, 300, 3), np.uint8)
    image[:] = colour
    show(name, image)

def show_matplot(img):
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    
def convert_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)