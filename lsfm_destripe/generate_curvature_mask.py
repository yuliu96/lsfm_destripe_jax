import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure
import skimage

def compute_curvature(point, i, contour, window_size):
    # Compute the curvature using polynomial fitting in a local coordinate system

    # Extract neighboring edge points
    start = max(0, i - window_size // 2)
    end = min(len(contour), i + window_size // 2 + 1)
    neighborhood = contour[start:end]

    # Extract x and y coordinates from the neighborhood
    x_neighborhood = neighborhood[:, 1]
    y_neighborhood = neighborhood[:, 0]

    # Compute the tangent direction over the entire neighborhood and rotate the points
    tangent_direction_original = np.arctan2(np.gradient(y_neighborhood), np.gradient(x_neighborhood))
    #tangent_direction_original[:] = tangent_direction_original[len(tangent_direction_original)//2]
    tangent_direction_original.fill(tangent_direction_original[len(tangent_direction_original)//2])
    # Translate the neighborhood points to the central point
    translated_x = x_neighborhood - point[1]
    translated_y = y_neighborhood - point[0]

    # Apply rotation to the translated neighborhood points
    # We have to rotate the points to be able to compute the curvature independent of the local orientation of the curve
    rotated_x = translated_x * np.cos(-tangent_direction_original) - translated_y * np.sin(-tangent_direction_original)
    rotated_y = translated_x * np.sin(-tangent_direction_original) + translated_y * np.cos(-tangent_direction_original)

    # Fit a polynomial of degree 2 to the rotated coordinates
    coeffs = np.polyfit(rotated_x, rotated_y, 2)


    # You can compute the curvature using the formula: curvature = |d2y/dx2| / (1 + (dy/dx)^2)^(3/2)
    # dy_dx = np.polyval(np.polyder(coeffs), rotated_x)
    # d2y_dx2 = np.polyval(np.polyder(coeffs, 2), rotated_x)
    # curvature = np.abs(d2y_dx2) / np.power(1 + np.power(dy_dx, 2), 1.5)

    # We compute the 2nd derivative in order to determine whether the curve at the certain point is convex or concave
    curvature = np.polyval(np.polyder(coeffs, 2), rotated_x)

    # Return the mean curvature for the central point
    return np.mean(curvature)

def compute_curvature(contour, window_size):
    m, n = contour.shape
    m = m-window_size+1
    neighborhood = skimage.util.view_as_windows(contour, (window_size, 1)).reshape(m, n, -1)
    x_neighborhood = neighborhood[:, 1, :]
    y_neighborhood = neighborhood[:, 0, :]
    tangent_direction_original = np.arctan2(np.gradient(y_neighborhood, axis = -1), np.gradient(x_neighborhood, axis = -1))
    tangent_direction_original = tangent_direction_original[:, window_size//2:window_size//2+1].repeat(window_size, axis=-1)
    translated_x = x_neighborhood - contour[window_size//2:-window_size//2+1, 1][:, None]
    translated_y = y_neighborhood - contour[window_size//2:-window_size//2+1, 0][:, None]
    rotated_x = translated_x * np.cos(-tangent_direction_original) - translated_y * np.sin(-tangent_direction_original)
    rotated_y = translated_x * np.sin(-tangent_direction_original) + translated_y * np.cos(-tangent_direction_original)
    r = []
    for i in range(m):
        coeffs = np.polyfit(rotated_x[i], rotated_y[i], 2)
        curvature = np.polyval(np.polyder(coeffs, 2), rotated_x[i])
        r.append(np.mean(curvature))
    return np.array(r)

def compute_curvature_profile(mask, min_contour_length, window_size_ratio):
    contours = measure.find_contours(mask, 0.5)

    # Initialize arrays to store the curvature information for each edge pixel
    curvature_values = []
    edge_pixels = []
    for contour in contours:
        # Iterate over each point in the contour
        if contour.shape[0] > min_contour_length:
            window_size = int(contour.shape[0]/window_size_ratio)
            curvature = compute_curvature(contour, window_size)
            curvature[curvature>0] = 1
            curvature[curvature<=0] = -1
            edge_pixels.append(contour[window_size//2:-window_size//2+1, :])
            curvature_values.append(curvature)

    # Convert lists to numpy arrays for further processing
    curvature_values = np.concatenate(curvature_values)
    edge_pixels = np.concatenate(edge_pixels)

    return edge_pixels, curvature_values

def plot_edges_with_curvature(mask, min_contour_length, window_size_ratio):
    edge_pixels, curvature_values = compute_curvature_profile(mask, min_contour_length, window_size_ratio)
    mask = np.zeros_like(mask)
    mask[edge_pixels[np.abs(curvature_values)>0.5, 0].astype(int), edge_pixels[np.abs(curvature_values)>0.5, 1].astype(int)] = 1
    return mask


def generate_mask(mask,min_contour_length = 20,window_size_ratio = 5):
    img_blur = (mask - mask.min()) / (mask.max()-mask.min()) *255
    ot = skimage.filters.threshold_otsu(img_blur)
    edges = cv2.Canny(image=img_blur.astype(np.uint8), threshold1=ot, threshold2=ot*2) # Canny Edge Detection
    mask_de = plot_edges_with_curvature(edges, min_contour_length, window_size_ratio)
    return mask_de