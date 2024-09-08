import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates


# Load the image
image = plt.imread('face.jpg')

center_y_offset = 100 # 原影像的polar coordinate 中心y



# Get image dimensions
ny, nx = image.shape[:2]

# Define center
cx, cy = nx/2, ny + center_y_offset

# Create meshgrid of coordinates
x = np.linspace(0, nx-1, nx)
y = np.linspace(0, ny-1, ny)
xx, yy = np.meshgrid(x, y)

# Calculate polar coordinates
r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
theta = np.arctan2(yy - cy, xx - cx)

r0 = 300 #新影像的圓柱半徑
r1 = ny*2 #新影像的高度

# Adjust the radius r to create a new transformation
newR = r0 + (ny - yy) / ny * r1  # 適用於每個點

# Define newTheta and r1 for mapping
newTheta = np.tile(np.linspace(-np.pi/2, np.pi/2, nx), (ny, 1))


# Map polar coordinates back to Cartesian for interpolation
xx_polar = newR * np.cos(newTheta) 
yy_polar = newR * np.sin(newTheta)

yy_polar = -np.min(yy_polar) + yy_polar

#plt.scatter(np.arange(nx), xx_polar[100], s=0.01)  # `s=1` to set the marker size


# Determine the size of the new image based on the transformed coordinates
min_x, max_x = np.min(xx_polar), np.max(xx_polar)
min_y, max_y = np.min(yy_polar), np.max(yy_polar)


# Calculate the size of the new image
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


'''
# Populate the new image with pixel values from the original image
for i in range(ny):
    for j in range(nx):
        for k in range(image.shape[2]):
            x = xx_polar_int[i][j]
            y = yy_polar_int[i][j]
            if 0 <= x < new_width and 0 <= y < new_height:  # Ensure indices are within bounds
                newImage[y][x][k] = image[i][j][k]  # Note: y and x are flipped for row, column indexing
'''


# Display the transformed image
plt.imshow(newImage)
plt.axis('off')  # Hide the axis
#plt.savefig('output.png', dpi=150, bbox_inches='tight', pad_inches=0, transparent=False)
plt.savefig('output.tif', bbox_inches='tight', pad_inches=0, transparent=False, dpi=300, compression=None)

plt.show()


