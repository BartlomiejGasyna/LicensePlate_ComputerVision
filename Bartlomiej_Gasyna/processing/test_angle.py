import cv2
import numpy as np

# Load the image
image = cv2.imread('/home/gasyna/RiSA_S1/SW/ImageProcessingCourse/project/2023/DATASETS/tablice/GS64165.jpg')

# Define the rotation angle in degrees
angle_positive = 10
angle_negative = -10

# Calculate the rotation matrix
center = (image.shape[1] // 2, image.shape[0] // 2)
rotation_matrix_positive = cv2.getRotationMatrix2D(center, angle_positive, 1.0)
rotation_matrix_negative = cv2.getRotationMatrix2D(center, angle_negative, 1.0)

# Apply the rotation using warpAffine
rotated_image_positive = cv2.warpAffine(image, rotation_matrix_positive, (image.shape[1], image.shape[0]))
rotated_image_negative = cv2.warpAffine(image, rotation_matrix_negative, (image.shape[1], image.shape[0]))

# Display the original and rotated images
cv2.imshow('Original Image', image)
cv2.imshow('Rotated Image (+10 degrees)', rotated_image_positive)
cv2.imshow('Rotated Image (-10 degrees)', rotated_image_negative)
cv2.waitKey(0)
cv2.destroyAllWindows()