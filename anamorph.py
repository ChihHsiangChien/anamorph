import numpy as np
import cv2

def anamorphWithoutInterpolation(image):
    ny, nx = image.shape[:2]
    center_y_offset = 100
    cx, cy = nx / 2, ny + center_y_offset

    # Define the radius and height for the new image
    r0 = nx
    r1 = ny * 2

    # Create meshgrid for coordinates
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)

    # Calculate polar coordinates    
    theta = np.arctan2(yy - cy, xx - cx)


    # Define newTheta and r1 for mapping
    newTheta = np.tile(np.linspace(-np.pi/2, np.pi/2, nx), (ny, 1))


    # Define new radius
    newR = r0 + (ny - yy) / ny * r1

    # Map polar coordinates back to Cartesian
    xx_polar = newR * np.cos(newTheta)
    yy_polar = newR * np.sin(newTheta)
    yy_polar = -np.min(yy_polar) + yy_polar

    # Calculate new dimensions
    min_x, max_x = np.min(xx_polar), np.max(xx_polar)
    min_y, max_y = np.min(yy_polar), np.max(yy_polar)
    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))

    # Initialize the new image with the calculated size
    newImage = np.full((new_height, new_width, image.shape[2]), 255, dtype=image.dtype)


    # Adjust the polar coordinates to the new image's coordinate system
    xx_polar_int = np.round(xx_polar - min_x).astype(int)
    yy_polar_int = np.round(yy_polar - min_y).astype(int)

    # Vectorized operation to populate the new image
    valid_indices = (xx_polar_int >= 0) & (xx_polar_int < new_width) & (yy_polar_int >= 0) & (yy_polar_int < new_height)
    newImage[yy_polar_int[valid_indices], xx_polar_int[valid_indices]] = image[valid_indices]
    return newImage



def anamorphWithoutMask(f):
    ny, nx = f.shape[:2]
    # Define the radius and height for the new image
    r0 = 300
    r1 = ny * 2

    # Create meshgrid for coordinates
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)

    r = r0 + r1 * (ny - yy - 1)/(ny - 1)
    theta = -np.pi/2 + np.pi * (nx - xx - 1)/(nx - 1)


    # Calculate new dimensions
    new_width  = 2 *(r0 + r1)
    new_height = 1 *(r0 + r1 )

    # Create new mapping matrices
    map_x, map_y = np.meshgrid(np.arange(new_width), np.arange(new_height))

    x0 = r1 + r0
    y0 = r1 + r0
    

    # 計算 r
    dx = map_x - x0
    dy = map_y - y0
    r = np.sqrt(dx**2 + dy**2)

    # 計算 theta
    theta = np.arctan2(dy, dx)

    # 計算 map_x
    map_x_new = nx - 1 - (nx - 1) * (theta + np.pi) / (np.pi)

    # 計算 map_y
    map_y_new = ny - 1 - (ny - 1) * (r - r0) / r1

    # Clip values to ensure they are within the valid range
    map_x_new = np.clip(map_x_new, 0, nx - 1).astype(np.float32)
    map_y_new = np.clip(map_y_new, 0, ny - 1).astype(np.float32)

    # Apply the remap transformation
    g = cv2.remap(f, map_x_new, map_y_new, interpolation=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    return g


def anamorph(f, start_radian  = np.pi,  spread_radian = 2 * np.pi):
    ny, nx = f.shape[:2]
    # Define the radius and height for the new image
    r0 = ny
    r1 = ny

    # Calculate new dimensions
    new_width = 2 * (r0 + r1)
    new_height = 2 *(r0 + r1)

    # Create meshgrid for coordinates
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)

    # Create new mapping matrices
    map_x, map_y = np.meshgrid(np.arange(new_width), np.arange(new_height))

    x0 = r1 + r0
    y0 = r1 + r0

    # Calculate r and theta
    dx = map_x - x0
    dy = map_y - y0
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)


    map_x_new = nx - 1 - (nx - 1) * (theta + start_radian) / spread_radian
    map_y_new = ny - 1 - (ny - 1) * (r - r0) / r1
    
    # Create masks for valid coordinates before clipping
    valid_x_mask = (map_x_new >= 0) & (map_x_new < nx)
    valid_y_mask = (map_y_new >= 0) & (map_y_new < ny)
    valid_mask = valid_x_mask & valid_y_mask

    # Initialize the output image with a white background
    g = np.full((new_height, new_width, 3), (255, 255, 255), dtype=np.uint8)

    # Initialize the remapped image
    temp = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Apply the remap transformation to the entire image
    temp = cv2.remap(f, map_x_new.astype(np.float32), map_y_new.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    # Apply the mask to combine the results
    g[valid_mask] = temp[valid_mask]

    return g


# Read input image
filename = 'gridWithPoint.png'
inputImg = cv2.imread(filename)
#outputImg = anamorph(inputImg)
#start_radian 從哪個方向開始
outputImg = anamorph(inputImg, start_radian = 0, spread_radian = 0.5*np.pi)

#outputImg = anamorphWithoutInterpolation(inputImg)
cv2.imwrite("output.jpg", outputImg)
