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
    cv2.createTrackbar('blur_main', 'image', 3, 20, nothing)

    cv2.createTrackbar('blur1', 'image', 3, 50, nothing)
    cv2.createTrackbar('blur2', 'image', 10, 50, nothing)

    cv2.createTrackbar('brightness', 'image', 28, 255, nothing)
    cv2.createTrackbar('contrast', 'image', 113, 130, nothing)

    cv2.createTrackbar('sigma0', 'image', 5, 100, nothing)
    cv2.createTrackbar('sigma1', 'image', 11, 100, nothing)
    cv2.createTrackbar('sigma2', 'image', 14, 100, nothing)
    
    # resized = cv2.resize(image, (640, 400), cv2.INTER_AREA)

    if image.shape[0] > image.shape[1]:
        dstx, dsty = 720, 960
    else:
        dstx, dsty = 960, 720

    resized = cv2.resize(image, (dstx, dsty), cv2.INTER_AREA)

    
    while True:
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        blur_main_c = cv2.getTrackbarPos('blur_main', 'image') * 2 + 1

        blur1_c = cv2.getTrackbarPos('blur1', 'image') * 2 + 1
        blur2_c = cv2.getTrackbarPos('blur2', 'image') * 2 + 1

        brightness = cv2.getTrackbarPos('brightness', 'image')
        contrast = cv2.getTrackbarPos('contrast', 'image')

        sigma0 = cv2.getTrackbarPos('sigma0', 'image') / 10.0
        sigma1 = cv2.getTrackbarPos('sigma1', 'image') / 10.0
        sigma2 = cv2.getTrackbarPos('sigma2', 'image') / 10.0

        gray = apply_brightness_contrast(gray, brightness, contrast)
        # blur = cv2.GaussianBlur(adjusted, (blur_main_c, blur_main_c), 0)
        # blur1 = cv2.GaussianBlur(blur, (blur1_c, blur1_c), 0)
        # blur2 = cv2.GaussianBlur(blur, (blur2_c, blur2_c), 0)

        width=0 
        height=0

        start_x=0 
        start_y=0
        end_x=0 
        end_y=0

        img_cpy = resized.copy()
        gw, gs, gw1, gs1, gw2, gs2 = (blur_main_c, sigma0, blur1_c, sigma1, blur2_c, sigma2)

   

        img_blur = cv2.GaussianBlur(gray, (gw, gw), gs)
        g1 = cv2.GaussianBlur(img_blur, (gw1, gw1), gs1)
        g2 = cv2.GaussianBlur(img_blur, (gw2, gw2), gs2)
        ret, thg = cv2.threshold(g2-g1, 127, 255, cv2.THRESH_BINARY)


        cv2.imshow('canny', thg)
        contours, hier = cv2.findContours(thg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        for i in range(len(contours)):
            if hier[0][i][2] == -1:
                continue
            
            x ,y, w, h = cv2.boundingRect(contours[i])
            a=w*h    
            aspectRatio = float(w)/h
            # if  aspectRatio >= 2 and a > 20000:          
            if  aspectRatio >= 2:          
                approx = cv2.approxPolyDP(contours[i], 0.05* cv2.arcLength(contours[i], True), True)
                if len(approx) == 4 and x>12  :
                    width=w
                    height=h   
                    start_x=x
                    start_y=y
                    end_x=start_x+width
                    end_y=start_y+height 
                    color = (0, 0, 255)
                    if a < 20_000:
                        color = (220, 220, 220)     
                    cv2.rectangle(img_cpy, (start_x,start_y), (end_x,end_y), color, 3)
                    cv2.putText(img_cpy, "rectangle "+str(x)+" , " + str(y-5), (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                
        cv2.imshow('image', img_cpy)


        key = cv2.waitKey(10)
        if key == ord('q'):
            cv2.destroyAllWindows()
            quit()
        if key == ord(' '):
            return 'PO12345'

