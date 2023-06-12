# License plate recognition

This project is the result of work carried out as part of the "Computer Vision Systems" course. The task involved recognizing license plate numbers in provided images.

This repository utilizes OpenCV and TensorFlow for license plate recognition. It includes various functions for image processing and neural network training and inference.

## Project structure
    ├── Bartłomiej_Gasyna.py    - main script, launches recognition
    ├── processing
    │   ├── charModel0.h5       - character recognition Tensorflow model
    │   ├── classes.npy         - tf labels saved to .npy for easy load
    │   ├── modelTest.py        - model inference / test
    │   ├── score.py            - % score (correct characters / sum of all characters)
    │   └── utils.py            - image processing handling
    └── results.json            - output results file for comparison, score calculation

### `main()` - Bartłomiej_Gasyna.py

The main() function processes a set of images and saves the results in a JSON file. It takes an input directory and an output file as command-line arguments, reads each image, performs some processing, and stores the results in a dictionary. Finally, it writes the dictionary to the output file as a JSON object.

## Plate recognition


The license plate recognition process involves the following steps:

1. Resizing the input image to a constant size and converting it to grayscale.
2. Applying changes in contrast and brightness to enhance image features.
3. Applying Gaussian blur to suppress noise.
4. Subtracting Gaussian blurs `G1` and `G2` with different kernels from each other `(G2 - G1)`, resulting in a clear view of edges.
5. Utilizing `cv2.findContours()` and `cv2.drawContours()` to find and draw contours on the image.
6. Checking each contour to determine if it meets the requirements for a rectangle with correct aspect ratio and area size. If so, a rectangle is found.



## `transform_corners(plate)` - Transformed Plate Image

Transforming the raw plate image by applying perspective transform using the four corners of the rectangle. This results in a straight view of the plate.

The transform_corners() function takes a raw plate image as input. It applies thresholding and finds contours using the cv2.RETR_EXTERNAL flag. Since the objective is to find the plate, which is the largest object in the computed rectangle, the function selects the contour with the largest area as the plate's contour.

Next, the function uses the Harris method to find corners. Only the four contours closest to the rectangle's edges (RMS) are selected as corner points.

Finally, a perspective transform is performed on the rectangle using these four corners, resulting in a straight view of the plate.

## `extract_letters(transformed) - Extracting Plate Numbers`

The extract_letters() function first applies Gaussian blur and thresholding to the transformed plate image. It then uses cv2.findContours() with the cv2.RETR_CCOMP flag to find contours on the image. The contour representing the plate's outer boundary is removed. Contours close to the edges are filtered based on their contour centers.


    M = cv2.moments(contour)
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    ...

Contours with very small or large area sizes are also discarded. The remaining contours, which have the right size and location, may still have holes in characters like 0, 8, and 9. To fix this, the function fills in black for all contours and then selectively fills in white only for smaller contours. This ensures clear images of the character contours.

Bounding boxes are computed for these contours, including a 5px padding. The bounding boxes are sorted from left to right and passed as input to the neural network. Each bounding box is processed by the network, resulting in character recognition.

If a frame is not found or the letters on the frame are not interpreted, the algorithm will rerun with different parameter sets, such as contrast, main Gaussian blur value, and rotation, to improve the recognition results.
