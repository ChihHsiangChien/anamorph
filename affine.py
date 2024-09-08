import cv2
import numpy as np

def scaleImage(f, scale_x=1, scale_y=1):
   
    # Get image dimensions
    height, width = f.shape[:2]

    # Calculate new dimensions
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)

    # Create new mapping matrices
    map_x, map_y = np.meshgrid(np.arange(new_width), np.arange(new_height))

    # Calculate inverse mapping
    map_x_new = map_x / scale_x
    map_y_new = map_y / scale_y

    # Clip values to be within original image dimensions
    map_x_new = np.clip(map_x_new, 0, width - 1).astype(np.float32)
    map_y_new = np.clip(map_y_new, 0, height - 1).astype(np.float32)

    # Perform remapping
    g = cv2.remap(f, map_x_new, map_y_new, interpolation=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return g

def translateImage(f, tx, ty):
    height, width = f.shape[:2]
    # Create the translation matrix
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    # Perform the translation
    g = cv2.warpAffine(f, translation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return g

def translateImage2(f, tx, ty):
    height, width = f.shape[:2]
    new_width = int(width  + np.abs(tx))
    new_height = int(height + np.abs(ty))
    # Create new mapping matrices
    map_x, map_y = np.meshgrid(np.arange(new_width), np.arange(new_height))

    # Calculate inverse mapping
    map_x_new = map_x - tx
    map_y_new = map_y - ty

    # Clip values to be within original image dimensions
    map_x_new = np.clip(map_x_new, 0, width - 1).astype(np.float32)
    map_y_new = np.clip(map_y_new, 0, height - 1).astype(np.float32)

    # Perform remapping
    g = cv2.remap(f, map_x_new, map_y_new, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return g


def rotationImage(f, theta=90):
    radian = np.deg2rad(theta)  # Convert degrees to radians

    # Get image dimensions
    height, width = f.shape[:2]

    center_x = width / 2
    center_y = height / 2

    # Calculate new dimensions
    new_width = int(np.abs(width * np.cos(radian)) + np.abs(height * np.sin(radian)))
    new_height = int(np.abs(height * np.cos(radian)) + np.abs(width * np.sin(radian)))

    # Create new mapping matrices
    map_x, map_y = np.meshgrid(np.arange(new_width), np.arange(new_height))

    # Center of the new image
    new_center_x = new_width / 2
    new_center_y = new_height / 2
    
    # Calculate inverse mapping
    map_x_new = (map_x - new_center_x) * np.cos(-radian) - (map_y - new_center_y) * np.sin(-radian) + center_x
    map_y_new = (map_x - new_center_x) * np.sin(-radian) + (map_y - new_center_y) * np.cos(-radian) + center_y

    # Clip values to be within original image dimensions
    map_x_new = np.clip(map_x_new, 0, width - 1).astype(np.float32)
    map_y_new = np.clip(map_y_new, 0, height - 1).astype(np.float32)

    # Perform remapping
    g = cv2.remap(f, map_x_new, map_y_new, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return g


# Read input image
filename = 'face.jpg'
inputImage = cv2.imread(filename)
#outputImg = scaleImage(inputImage, scale_x=0.5, scale_y=2)
outputImg = rotationImage(inputImage,45)
#outputImg = translateImage2(inputImage,-300,0)

# Display output image
cv2.imshow("outputImage", outputImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output image if needed
# cv2.imwrite('output.jpg', outputImg)
