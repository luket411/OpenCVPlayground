# https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
from sys import path
from os import path as ospath
path.append(f'{ospath.dirname(__file__)}/..')
import numpy as np
import cv2
from util import show_matplot as show
from grey_out_pixel import main as get_image_board_outline
from coordinates_system import Line, construct_line_from_points
import matplotlib.pyplot as plt
from line_classification import find_lines, normalise_infs_nans


# img_file = "assets/IMG_4281.jpg"
# img_file = "assets\PXL_20220121_220215270.jpg"
img_file = "assets/clean_board.jpg"


# threshold_max = 0
# linLength_max = 0
# current_max = 0

# for threshold in range(0,500,50):
    # for minLineLength in range(0,100,2):

    #     lines = hlp(threshold, minLineLength)
        
    #     if lines is not None:
    #         if lines.shape[0] > current_max:
    #             current_max = lines.shape[0]
    #             threshold_max = threshold
    #             linLength_max = minLineLength
    #             print(f"Current max shape: {current_max} @ threshold={threshold}, minLineLength={minLineLength}")
    #         print(f"lines.shape:{lines.shape} @ threshold={threshold}, minLineLength={minLineLength}")


        # print( f"lines.shape:{lines.shape}" if lines is not None else "lines.shape:None")

def find_corners(img_file, show_output=False, debug_print=False):
    img = blur_and_colour_shift(img_file, show_output)
    canny_edges = run_canny(img, show_output=show_output)
    line_points = run_hough(canny_edges, debug_print=debug_print)
    line_details = gather_line_details(line_points)
    norm_grads = normalize_grads(line_details[0])
    lines, labels = classify_lines(norm_grads, *line_details)
    edges = define_edges(lines, labels, debug_print=debug_print)
    corners = find_intersection_points(edges)


    if show_output:
        colours = [[0,200,0],[200,0,0],[0,0,200],[3, 252, 244]]
        output = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB)
        if show_output:
            for i, edge in enumerate(edges):
                output = edge.plot(output, colours[i])
    
        show(output)

    return corners


def blur_and_colour_shift(img_file, show_output):
    img = cv2.imread(img_file, 1)
    if show_output:
        show(img)

    # Heavily blur the image, this reduces the amount of edges detected on the board and background
    img = cv2.GaussianBlur(img, (15,15),45)
    if show_output:
        show(img)

    # Get the middle 15-85% of a selected white region and remove it
    img = get_image_board_outline(img, 15)
    if show_output:
        show(img)

    # Convert the image to grey
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if show_output:
        show(img)
    return img

def run_canny(img, threshold1=210, threshold2=255, edges=1, apertureSize=3, L2gradient=None, show_output=False):
    canny_edges = cv2.Canny(
        image=img,
        threshold1=threshold1, # Only detect sharp changes
        threshold2=threshold2, # 
        edges=edges,
        apertureSize=apertureSize,
        L2gradient=L2gradient
    )
    
    if show_output:
        show(canny_edges)
    
    return canny_edges

def run_hough(image, threshold=180, minLineLength=20, debug_print=False):
    line_points = cv2.HoughLinesP(image=image,
                            rho=1,
                            theta=np.pi/180,
                            threshold=threshold,
                            minLineLength=minLineLength
                            )

    if debug_print:
        print( f"lines.shape:{line_points.shape}" if line_points is not None else "lines.shape:None")

    if line_points is None:
        raise Exception("No Lines found")
    return line_points

def gather_line_details(line_points, debug_print=False):

    grads = []
    y_intercepts = []
    x_intercepts = []
    points = []

    for i in range(0, len(line_points)):
        l = line_points[i][0]
        
        line = construct_line_from_points(l[0], l[1], l[2], l[3])
        
        grads.append(line.grad)
        y_intercepts.append(line.y_intercept)
        x_intercepts.append(line.x_intercept)
        points.append([l[0], l[1], l[2], l[3]])

    return grads, y_intercepts, x_intercepts, points

def normalize_grads(grads):
    non_inf_max_grad = np.max([abs(grad) for grad in grads if not(np.isinf(grad) or np.isnan(grad))])
    norm_grads = np.absolute(normalise_infs_nans(grads, non_inf_max_grad, 0))
    
    return norm_grads

def classify_lines(norm_grads, grads, y_intercepts, x_intercepts, points):
    outliers = []
    vertical_lines = []
    horizontal_lines = []
    
    for i, grad in enumerate(norm_grads):
        if grad > 0.9:
            vertical_lines.append([grads[i], y_intercepts[i], x_intercepts[i], points[i]])
        elif grad < 0.001:
            horizontal_lines.append([grads[i], y_intercepts[i], x_intercepts[i], points[i]])
        else:
            outliers.append([grads[i], y_intercepts[i], x_intercepts[i], points[i]])

    vertical_lines = np.array(vertical_lines, dtype=object)
    horizontal_lines = np.array(horizontal_lines, dtype=object)

    vertical_intercept_avg = np.average(vertical_lines[:,2])
    vertical_labels = (vertical_lines[:,2] > vertical_intercept_avg).astype(np.int8)
    
    horizontal_intercept_avg = np.average(horizontal_lines[:,1])
    horizontal_labels = (horizontal_lines[:,1] > horizontal_intercept_avg).astype(np.int8)
    horizontal_labels += 2

    lines = np.concatenate((vertical_lines, horizontal_lines), axis=0)
    labels = np.concatenate((vertical_labels, horizontal_labels), axis=0)

    return lines, labels
    
def define_edges(lines, labels, debug_print=False):
    edge_lines_details = [
        [
            [],
            [],
            []
        ],
        [
            [],
            [],
            []
        ],
        [
            [],
            [],
            []
        ],
        [
            [],
            [],
            []
        ]
    ]
    
    for i, line_obj in enumerate(lines):
        line_class = labels[i]
        edge_lines_details[line_class][0].append(line_obj[0])
        edge_lines_details[line_class][1].append(line_obj[1])
        edge_lines_details[line_class][2].append(line_obj[2])
        
        l = line_obj[3]       
        if debug_print:
            print(f"Line{i}: ({l[0]},{l[1]}) -> ({l[2]}, {l[3]}) (Grad: {line_obj[0]}, y_intercept: {line_obj[1]}, x_intercept: {line_obj[2]}, line_class: {line_class})")

        
    edges = []    
    
    for line_details in edge_lines_details:
        avg_grads = np.average(line_details[0])
        avg_y_intercepts = np.average(line_details[1])
        avg_x_intercepts = np.average(line_details[2])
        edge = Line(avg_grads, avg_y_intercepts, avg_x_intercepts)
        edges.append(edge)

    if debug_print:
        print(edges)

    return edges

def find_intersection_points(edges):
    intersection1 = np.array(edges[0].find_intersection(edges[2]))
    intersection2 = np.array(edges[0].find_intersection(edges[3]))
    intersection3 = np.array(edges[1].find_intersection(edges[2]))
    intersection4 = np.array(edges[1].find_intersection(edges[3]))
    
    return(intersection_points := [intersection1, intersection2, intersection3, intersection4])

if __name__ == "__main__":
    print(find_corners(img_file, True, True))
        