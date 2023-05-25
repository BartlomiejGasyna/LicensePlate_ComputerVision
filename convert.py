import cv2
import os
import time
# Directory path containing the images
directory = 'dataset/'
directory_new = 'dataset_new/'
# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        # Read the image
        image_path = os.path.join(directory, filename)
        image = cv2.imread(image_path)
        
        # Convert the image to JPEG format and save it
        new_filename = str(int(time.time()*1000000)) + '.jpg'
        new_image_path = os.path.join(directory_new, new_filename)
        cv2.imwrite(new_image_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        print(f"Converted {filename} to JPEG format.")

