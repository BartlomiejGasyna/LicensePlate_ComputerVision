import numpy as np
import cv2

def nothing(x):
    pass

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

# def areaFilter(minArea, inputImage):
#     # Perform an area filter on the binary blobs:
#     componentsNumber, labeledImage, componentStats, componentCentroids = \
#         cv2.connectedComponentsWithStats(inputImage, connectivity=4)

#     # Get the indices/labels of the remaining components based on the area stat
#     # (skip the background component at index 0)
#     remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

#     # Filter the labeled pixels based on the remaining labels,
#     # assign pixel intensity to 255 (uint8) for the remaining pixels
#     filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

#     return filteredImage

# def CMYK():
    ##### CMYK
    imgFloat = apply_brightness_contrast(resized,contrast=100)
    cv2.imshow('contrast', imgFloat)
    imgFloat = resized.astype(np.float) / 255.
    # Calculate channel K:
    kChannel = 1 - np.max(imgFloat, axis=2)
    # Convert back to uint 8:
    kChannel = (255 * kChannel).astype(np.uint8)

    # Threshold image:
    binaryThresh = 190
    _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255, cv2.THRESH_BINARY)

    minArea = 100
    binaryImage = areaFilter(minArea, binaryImage)
    # Use a little bit of morphology to clean the mask:
    # Set kernel (structuring element) size:
    kernelSize = 3
    # Set morph operation iterations:
    opIterations = 2
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform closing:
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)

    cv2.imshow("binaryImage [closed]", binaryImage)


    cv2.waitKey()
    ##### CMYK

# trackbars = {'blur_main': 0, 'blur1': 0, 'blur2': 0, 'brightness': 0, 'contrast': 0}

# trackbars['blur_main'] = blur_main_c
# trackbars['blur1'] = blur1_c
# trackbars['blur2'] = blur2_c
# trackbars['brightness'] = brightness
# trackbars['contrast'] = contrast

# cv2.setTrackbarPos('blur_main', 'image', trackbars['blur_main'])
# cv2.setTrackbarPos('blur1', 'image', trackbars['blur1'])
# cv2.setTrackbarPos('blur2', 'image', trackbars['blur2'])

# cv2.setTrackbarPos('brightness', 'image', trackbars['brightness'])
# cv2.setTrackbarPos('contrast', 'image', trackbars['contrast'])
def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    
    cv2.namedWindow('image')
    # creating trackbars for red color change
    cv2.createTrackbar('blur_main', 'image', 2, 20, nothing)

    cv2.createTrackbar('blur1', 'image', 7, 50, nothing)
    cv2.createTrackbar('blur2', 'image', 6, 50, nothing)

    cv2.createTrackbar('brightness', 'image', 0, 255, nothing)
    cv2.createTrackbar('contrast', 'image', 100, 130, nothing)
    
    # cv2.createTrackbar('canny_th1', 'image', 0, 255, nothing)
    # cv2.createTrackbar('canny_th2', 'image', 0, 255, nothing)
    resized = cv2.resize(image, (640, 400), cv2.INTER_AREA)

    
    while True:
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        blur_main_c = cv2.getTrackbarPos('blur_main', 'image') * 2 + 1

        blur1_c = cv2.getTrackbarPos('blur1', 'image') * 2 + 1
        blur2_c = cv2.getTrackbarPos('blur2', 'image') * 2 + 1

        brightness = cv2.getTrackbarPos('brightness', 'image')
        contrast = cv2.getTrackbarPos('contrast', 'image')

        # canny_th1 = cv2.getTrackbarPos('canny_th1', 'image')
        # canny_th2 = cv2.getTrackbarPos('canny_th2', 'image')



        adjusted = apply_brightness_contrast(gray, brightness, contrast)
        blur = cv2.GaussianBlur(adjusted, (blur_main_c, blur_main_c), 0)
        # blur = cv2.bilateralFilter(adjusted, 9, blur_main_c, blur_main_c)
        # _, blur = cv2.threshold(blur, otsu1, otsu2, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('after', blur)
        blur1 = cv2.GaussianBlur(blur, (blur1_c, blur1_c), 0)
        blur2 = cv2.GaussianBlur(blur, (blur2_c, blur2_c), 0)

        gray = blur1-blur2
        # blur = cv2.GaussianBlur(gray, (blur_main_c, blur_main_c), 0)

        # cv2.imshow('blur1-blur2', gray)

        # gray = cv2.bilateralFilter(gray, 9, blur1_c, blur2_c)
        # blur2 = cv2.bilateralFilter(gray, 9, blur2_c, blur1_c)

        # gray = blur1 - blur2
        # th = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,3)
        # th = apply_brightness_contrast(th, contrast=10)
        # cv2.imshow('th', th)
        # gray = cv2.bitwise_not(gray)
        # edges = cv2.bilateralFilter(gray, 15, 80, 80, cv2.BORDER_DEFAULT) 
        # gray = cv2.medianBlur(gray, 5)
        edges = cv2.Canny(gray, 80, 40, apertureSize=3)
    

        # get contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        image2 = resized.copy()
        cv2.drawContours(image2, contours, -1, (0,255,0), 3)
        cv2.imshow("Top 30 contours",image2)

        cv2.imshow("image with detected license plate", gray)


        cv2.imshow('edges', edges)

 
        key = cv2.waitKey(10)
        if key == ord('q'):
            cv2.destroyAllWindows()
            quit()
        if key == ord(' '):
            return 'PO12345'
