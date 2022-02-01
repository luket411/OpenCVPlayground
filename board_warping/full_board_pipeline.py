from sys import path
from os import getcwd, path as ospath
path.append(f'{ospath.dirname(__file__)}/..')
import numpy as np
import cv2
from util import show as show_cv2, show_matplot as show
from show_warped_board import transform_board, transform_size, desired_corners, annotate_fixed_city_points
from line_detection import find_corners
np.seterr(all="ignore")

if __name__ == "__main__":
    img_file = "assets/clean_board.jpg"
    img = cv2.imread(img_file, 1)
    show(img)
    source_corners = np.array(sorted(find_corners(img_file), key=lambda x: x.sum()), dtype=np.float32)
    target_corners = np.array(sorted(desired_corners, key=lambda x: x.sum()), dtype=np.float32)
    
    print(source_corners)
    print(target_corners)
    
    warped_board = transform_board(img, source_corners, target_corners, transform_size)
    warped_board = annotate_fixed_city_points(warped_board)
    show(warped_board)