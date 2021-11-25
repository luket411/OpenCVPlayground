import cv2
import numpy as np
from util import show, show_colour
from datarunner import run_all


file = 'assets/IMG_4281.jpg'

def get_average_colour(im):
    avg = [0,0,0]
    counter = 0
    for row in im:
        for pixel in row:
            counter += 1
            avg[0] += pixel[0]
            avg[1] += pixel[1]
            avg[2] += pixel[2]
    
    avg[0] //= counter
    avg[1] //= counter
    avg[2] //= counter
    
    return avg

def get_colour_percentile(im, n):
    avg = [0,0,0]
    avg[0] = np.percentile(im[:,:,0].flatten(), n)
    avg[1] = np.percentile(im[:,:,1].flatten(), n)
    avg[2] = np.percentile(im[:,:,2].flatten(), n)
    return avg

def find_colour(im, colour):
    colour = np.array(colour)
    mask = cv2.inRange(im, colour, colour)
    output = cv2.bitwise_and(im, im, mask = mask)
    show("Only Colour", output)

def find_colour_range(im, lower, upper):
    lower = np.array(lower)
    upper = np.array(upper)
    mask = cv2.inRange(im, lower, upper)
    output = cv2.bitwise_and(im, im, mask=mask)
    im = cv2.resize(im, (0, 0), fx=0.25, fy=0.25)
    output = cv2.resize(output, (0, 0), fx=0.25, fy=0.25)
    show("Range of Colour", np.hstack([im, output]))

@run_all
def find_colour_range_full(im, lower, upper):
    im = cv2.imread(im, 1)
    find_colour_range(im, lower, upper)


def sample_from_range(im, range):
    lower = get_colour_percentile(im, range)
    upper = get_colour_percentile(im, 100-range)

    show_colour(lower, "darker")
    show_colour(upper, "lighter")

    find_colour_range_full(lower = lower, upper = upper)
    



img = cv2.imread(file, 1)
# Yellow
yellow = img[1110:1222, 242:495]
# sample_from_range(yellow, 15)

# Green
green = img[1347:(1347+207), 3736:(3736+195)]
sample_from_range(green, 20)

# Blue
blue = img[2390:2390+186, 3059:3059+337]
# sample_from_range(blue, 20)

#Red
red = img[1798:1798+159, 139:139+255]
# sample_from_range(red, 25)

#Black
black = img[753:753+113, 3565:3565+215]
# sample_from_range(black, 25)