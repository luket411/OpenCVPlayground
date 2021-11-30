#Adapted from:
#https://towardsdatascience.com/image-processing-with-python-applying-homography-for-image-warping-84cd87d2108f

from util import show_matplot as show, convert_to_rgb
from datarunner import run_all, cities_loader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from skimage import transform
from skimage.io import imread as skimread

def warp_debug_prev(img, src, target):
    colour = 'red'
    fig, ax = plt.subplots(2,2, figsize=(20, 12))
    for im in src:
        ax[0][0].add_patch(Circle((im[0],im[1]), 20, facecolor = colour))

    ax[0][0].imshow(convert_to_rgb(img))
    ax[0][0].set_title('Original', fontsize=15)

    for im in target:
        ax[0][1].add_patch(Circle((im[0],im[1]), 20, facecolor = colour))
        ax[1][0].add_patch(Circle((im[0],im[1]), 20, facecolor = colour))
        # ax[1][1].add_patch(Circle((im[0],im[1]), 20, facecolor = colour))

    # transform_size = (img.shape[1], img.shape[0])
    transform_size = (3000, 2000)


    ax[0][1].imshow(np.ones(transform_size[::-1]))
    ax[0][1].set_title('Target', fontsize=15)


    M1 = cv2.getPerspectiveTransform(src, target)
    t1 = cv2.warpPerspective(img, M1, transform_size)

    ax[1][0].set_title('Warped 1 (OpenCV)', fontsize=15)
    ax[1][0].imshow(convert_to_rgb(t1))


    # img1 = skimread(file)

    # tform = transform.estimate_transform('projective', src, target)
    # t2 = transform.warp(img1, tform.inverse, mode = 'symmetric')

    # M2 = cv2.estimateAffine2D(imc.get_BAD(), dec.get_BAD())

    # t2 = cv2.warpAffine(img, M2[0], (4500, 3000))

    # ax[1][1].set_title('Warped 2 (Sci-kit Image)', fontsize=15)
    # ax[1][1].imshow(t2)

    ax[1][1].set_axis_off()

    plt.show()
    
def warp_debug(img, src, target):
    colour = 'red'
    fig, ax = plt.subplots(1,2, figsize=(16, 8))
    for im in src:
        ax[0].add_patch(Circle((im[0],im[1]), 20, facecolor = colour))

    ax[0].imshow(convert_to_rgb(img))
    ax[0].set_title('Original', fontsize=15)        
    
    for im in target:
        ax[1].add_patch(Circle((im[0],im[1]), 20, facecolor = colour))
    
    # transform_size = (img.shape[1], img.shape[0])
    transform_size = (3000,2000)

    M1 = cv2.getPerspectiveTransform(src, target)
    t1 = cv2.warpPerspective(img, M1, transform_size)

    ax[1].set_title('Warped 1 (OpenCV)', fontsize=15)
    ax[1].imshow(convert_to_rgb(t1))
    
    plt.show()
    
if __name__ == "__main__":
    file = 'assets/IMG_4281.jpg'
    file1 = 'assets/IMG_1036.jpg'
    img = cv2.imread(file1, 1)

    #####################
    # A---------------D #
    # |               | #
    # B---------------C #
    #####################

    # desired_corners = np.array([[500,500],
    #                    [500,2500],
    #                    [3500,2500],
    #                    [3500,500]], dtype='float32')

    corners_4281 = np.array([[830,700],[448,2174],[3582,2099],[3163,613]], dtype='float32')

    corners_1036 = np.array([[843.3,782.5],[243.5,2363.9],[3434,2814.7],[3333.1,960.5]], dtype='float32')

    desired_corners = np.array([[0,0],[0,2000],[3000,2000],[3000,0]], dtype='float32')
    
    cities = cities_loader()
    
    city_locations1 = np.array([[1386,1936],[1521,1693],[1738,1555],[1597,1446]], dtype='float32')
    desired_city_locations1 = np.array([cities.marseille, cities.zurich, cities.munchen, cities.frankfurt], dtype='float32')
    
    city_locations2 = np.array([[1167,889],[414,2146],[3237.4,2510],[3196,1321]], dtype='float32')
    desired_city_locations2 = np.array([cities.edinburgh, cities.lisboa, cities.erzurum, cities.moskva], dtype='float32')
    
    warp_debug(img, city_locations2, desired_city_locations2)