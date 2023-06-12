import numpy as np
import cv2
import math 
import processing.modelTest as modelTest
from tensorflow import keras
from keras.models import load_model


BLUR_MAIN = 7
BLUR1 = 9
BLUR2 = 21

BRIGHTNESS = 5
CONTRAST = 20

SIGMA0 = 3.5
SIGMA1 = 0.1
SIGMA2 = 1.1

SIZE_X = 960
SIZE_Y = 720

INTENSITY = 127


def nothing(x):
    pass

def auto_adjust_brightness_contrast(image, target_intensity = INTENSITY):
    # Calculate the mean pixel intensity
    mean_intensity = cv2.mean(image)[0]

    # Calculate the desired mean intensity (mid-gray)
    # target_intensity = 127
    # target_intensity = INTENSITY

    # Calculate the scaling factor for adjusting brightness and contrast
    alpha = target_intensity / mean_intensity

    # Apply the scaling factor to adjust brightness and contrast
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    return adjusted_image

def apply_brightness_contrast(input_img, brightness = BRIGHTNESS, contrast = CONTRAST):    
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


def transform_corners(image: np.ndarray) -> np.ndarray:
    transformed = image.copy()
    try:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    except:
        return transformed
    rt, gray = cv2.threshold(gray, 160, 255, cv2.THRESH_OTSU)
    
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Find the innermost contour
    inner_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            inner_contour = contour

    # Draw the innermost contour on the image
    # contours_img = cv2.drawContours(image, [inner_contour], 0, (0, 0, 255), 2)
    # cv2.imshow('inner_most', contours_img)
    # cv2.waitKey()

    dst = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    dst = cv2.drawContours(dst, [inner_contour], 0, (0, 0, 255), 2)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    
    image[dst>0.01*dst.max()]=[0,255,0]
    x0, x_end = 0, image.shape[0]
    y0, y_end = 0, image.shape[1]

    
    points = np.transpose(np.nonzero(dst>0.01*dst.max()))

    corners = np.float32([[x0, y0], [x_end, y0], [x0, y_end], [x_end, y_end]])

    x = 0
    y = 0
    Error = lambda p: math.sqrt((p[0]-x) ** 2 + (p[1]-y) ** 2)

    best_matches = []
    for corner in corners:
        x = corner[0]
        y = corner[1]
        
        values = list(map(Error, points))

        index = values.index(min(values))
        best_matches.append(points[index])
    

    for corner in best_matches:
        cv2.circle(image, (corner[1], corner[0]), 3, (255, 0, 0), 5)

    
    best_matches = np.float32(best_matches)
    corners = corners[:, [1, 0]]
    best_matches = best_matches[:, [1, 0]]
    matrix = cv2.getPerspectiveTransform(best_matches, corners)
    transformed = cv2.warpPerspective(transformed, matrix, (y_end, x_end))

    # Draw transformed image
    # cv2.imshow('corners', image)    
    # cv2.waitKey()

    # Draw transformed image
    # cv2.imshow('transformed', transformed)    
    # cv2.waitKey()


    return transformed

def extract_letters(plate: np.ndarray):
    try:
        plate = cv2.resize(plate, (1040, 228), cv2.INTER_AREA)
    except:
        return ''
    gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
    padding = 5
    new_height = gray.shape[0] + 2 * padding
    new_width = gray.shape[1] + 2 * padding
    padded_image = np.zeros((new_height, new_width), dtype=np.uint8)
    padded_image.fill(255)  # Fill with white color (255)

    gray = cv2.GaussianBlur(gray, (7, 7), 3.5)
    rt, gray = cv2.threshold(gray, 200, 255, cv2.THRESH_OTSU)

    # Calculate the starting position to paste the original image
    start_x = padding
    start_y = padding

    # Copy the original image onto the padded image at the specified position
    padded_image[start_y:start_y + gray.shape[0], start_x:start_x + gray.shape[1]] = gray
    copy = padded_image.copy()

    contours, hierarchy = cv2.findContours(padded_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    dst = np.zeros((padded_image.shape[0], padded_image.shape[1], 3), dtype=np.uint8)


    largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    
    contours = list(contours)
    del contours[largest_contour_index]
    

    letters =[]
    for idx, contour in enumerate(contours):
        M = cv2.moments(contour)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except:
            cX = 0
            cY = 0

        distX = 50
        distY = 50
        area = cv2.contourArea(contour)

        if area < 1000 or area > 25000:
            # print('deleted')
            continue
        
        if cX < distX or cX > dst.shape[1] - distX or cY < distY or cY > dst.shape[0] - distY:
            continue

        letters.append(idx)
    contours_filtered = [val for idx, val in enumerate(contours) if idx in letters ]


    cv2.fillPoly(dst, pts = contours_filtered, color=(255,255,255))

    for cnt in contours_filtered:
        if cv2.contourArea(cnt) < 3500:
            cv2.fillPoly(dst, pts = [cnt], color=(0,0,0))
            # print('filled artefacts')






    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('letters', dst)
    # cv2.waitKey(1000)
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    contours_filtered = []
    for i, contour in enumerate(contours):
        # Retrieve the hierarchy information for the current contour
        hierarchy_info = hierarchy[0][i]

        # Check if the contour is an external contour (no parent and no child)
        if hierarchy_info[3] == -1:
            contours_filtered.append(contour)

    contours = contours_filtered
    # contours_img = cv2.drawContours(dst, contours, -1, (0,0,0), 2)
    # cv2.imshow('letters_bboxes', contours_img)
    # cv2.waitKey()


    dst = cv2.bitwise_not(dst)

    boundRect =[]
    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True)
        rect = cv2.boundingRect(contours_poly)
        boundRect.append(rect)

    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    boundRect = sorted(boundRect, key = lambda box: box[0])

    # Draw bounding rects
    # Iterate over the boundingRect list
    # cv2.imshow('bounding_bboxes', dst)
    # cv2.waitKey()
    # for rect in boundRect:
    #     x, y, w, h = rect
    #     # Draw the rectangle on the image
    #     cv2.rectangle(dst, (x, y), (x-15 + w+15, y-15 + h+15), (0, 0, 255), 2)
    #     cv2.imshow('bounding_bboxes', dst)
    #     cv2.waitKey(330)


    license_plate = []
    
    for i in range(len(boundRect)):


        x, y, w, h = boundRect[i]
        letter = dst[max(y-5, 0):min(y+h+5, dst.shape[0]), max(x-5, 0):min(x+w+5, dst.shape[1])]
        
        # clear_letter = copy[max(y-5, 0):min(y+h+5, dst.shape[0]), max(x-5, 0):min(x+w+5, dst.shape[1])]

        model = load_model("processing/charModel0.h5")

        label = modelTest.inferImg(letter, model)
        # cv2.imshow(label, letter)
        
        # cv2.waitKey(800)

        # cv2.destroyAllWindows()


        # Numbers cannot be used in first part of registry plate # Individual tables not included
        changes1 = {'0': 'O',
                    '1': 'I',
                    '2': 'Z',
                    '4': 'A',
                    '5': 'Z',
                    '8': 'B'
                    }
        
        # Polish law regulates that letters B, D, I, O, Z cannot be used in second part of registry plate # Individual tables not included
        changes2 = {'B': '8', 
                    'D': '0', 
                    'I': '1',
                    'O': '0', 
                    'Z': '2'}
                    
        if i < 2 and label in changes1:
            label = changes1[label]
        elif i > 2 and label in changes2:
            label = changes2[label]

        license_plate.append(label)
        # import random
        # cv2.imwrite('extras/' + label + '_' + str(random.randint(0, 1000)) + '.jpg', letter)

    plane_number = ''.join(license_plate)
    # print('license plate: ', plane_number)

    return plane_number

    

def perform_processing(image: np.ndarray, contrast=CONTRAST, blur_c = BLUR_MAIN, padding = 0) -> str:
    brightness = BRIGHTNESS
    contrast_list = [10, 30, 40, 50, 80, 100, 120, 140, 10, 30, 40, 50, 80, 100, 120, 140]
    blur_list = [1, 1, 1, 1, 1, 1, 1, 1,7, 7, 7, 7, 7, 7, 7, 7]

#  bez:  0.9186602870813397
#  z:    0.9234449760765551
    if perform_processing.counter > 2 and perform_processing.counter < 5:
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # rotate our image by 45 degrees around the center of the image
        rot = 5
        if perform_processing.counter % 2:
            rot *=-1
        M = cv2.getRotationMatrix2D((cX, cY), rot, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

    perform_processing.counter += 1
    # print(f'image.shape: {image.shape}')

    # cv2.namedWindow('image')
    # # creating trackbars for red color change
    # cv2.createTrackbar('blur_main', 'image', 3, 20, nothing)

    # cv2.createTrackbar('blur1', 'image', 3, 50, nothing)
    # cv2.createTrackbar('blur2', 'image', 10, 50, nothing)

    # cv2.createTrackbar('brightness', 'image', 5, 255, nothing)
    # cv2.createTrackbar('contrast', 'image', 20, 130, nothing)

    # cv2.createTrackbar('sigma0', 'image', 35, 100, nothing)
    # cv2.createTrackbar('sigma1', 'image', 1, 100, nothing)
    # cv2.createTrackbar('sigma2', 'image', 11 , 100, nothing)
    

    image_raw_copy = image.copy()

    if image.shape[0] > image.shape[1]:
        dstx, dsty = SIZE_Y, SIZE_X
    else:
        dstx, dsty = SIZE_X, SIZE_Y

    resized = cv2.resize(image, (dstx, dsty), cv2.INTER_AREA)

    
    while True:
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # blur_main_c = cv2.getTrackbarPos('blur_main', 'image') * 2 + 1

        # blur1_c = cv2.getTrackbarPos('blur1', 'image') * 2 + 1
        # blur2_c = cv2.getTrackbarPos('blur2', 'image') * 2 + 1

        # brightness = cv2.getTrackbarPos('brightness', 'image')
        # contrast = cv2.getTrackbarPos('contrast', 'image')

        # sigma0 = cv2.getTrackbarPos('sigma0', 'image') / 10.0
        # sigma1 = cv2.getTrackbarPos('sigma1', 'image') / 10.0
        # sigma2 = cv2.getTrackbarPos('sigma2', 'image') / 10.0


        gray = apply_brightness_contrast(gray, brightness, contrast)

        img_cpy = resized.copy()
        gw, gs, gw1, gs1, gw2, gs2 = (blur_c, SIGMA0, BLUR1, SIGMA1, BLUR2, SIGMA2)

        img_blur = cv2.GaussianBlur(gray, (gw, gw), gs)
        g1 = cv2.GaussianBlur(img_blur, (gw1, gw1), gs1)
        g2 = cv2.GaussianBlur(img_blur, (gw2, gw2), gs2)
        # cv2.imshow('g2-g1', g2-g1)
        # cv2.waitKey()
        ret, thg = cv2.threshold(g2-g1, 160, 255, cv2.THRESH_OTSU)


        contours, hier = cv2.findContours(thg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        contours_img = np.zeros((dsty, dstx, 3), dtype=np.uint8)

        contours_img = cv2.drawContours(contours_img, contours, -1, (220, 220, 220), 1)
        # cv2.imshow('countours', contours_img)
        # cv2.waitKey()

        plate_number = 'PO12345'
        for i in range(len(contours)):
            if hier[0][i][2] == -1:
                continue
            
            x ,y, w, h = cv2.boundingRect(contours[i])
            a=w*h    
            aspectRatio = float(w)/h
            if  aspectRatio >= 2 and a > 20000:          
            # if  aspectRatio >= 2:  
            # if  aspectRatio >= 2:          
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

                    ex = padding
                    plate1 = img_cpy[ start_y-ex:end_y+ex, start_x-ex:end_x+ex].copy()
                    # plate1 = img_cpy[ start_y:end_y, start_x:end_x].copy()
                    plate2 = thg[ start_y:end_y, start_x:end_x].copy()

                    transformed = transform_corners(plate1)
                    plate_number = extract_letters(transformed)

                    if plate_number == "":
                        continue
                    # cv2.rectangle(img_cpy, (start_x,start_y), (end_x,end_y), color, 3)
                    # cv2.rectangle(gray, (start_x,start_y), (end_x,end_y), (255, 255, 255), 3)

                    # cv2.putText(img_cpy, "rectangle "+str(x)+" , " + str(y-5), (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    
                    if len(plate_number) > 5:
                        break
        
        if len(plate_number) < 4 or plate_number == 'PO12345':
            if perform_processing.counter > len(contrast_list)-1:
                perform_processing.counter = 0
                if plate_number == '':
                    plate_number = 'PO12345'
            else:
                contrast = contrast_list[perform_processing.counter]
                blur_c = blur_list[perform_processing.counter]
                # padding = padding_list[perform_processing.counter]
                # plate_number = perform_processing(image_raw_copy, contrast = contrast, blur_c = blur, padding = padding)
                plate_number = perform_processing(image_raw_copy, contrast = contrast, blur_c=blur_c)


        # cv2.imshow('01_plate_detected', gray)
        # cv2.waitKey()



        # key = cv2.waitKey(10)
        # if key == ord('q'):
        #     cv2.destroyAllWindows()
        #     quit()

        # cv2.waitKey(100)
        return plate_number

perform_processing.counter = 0