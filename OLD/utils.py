import numpy as np
import cv2
import math 
import time

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


def transform_corners(image: np.ndarray) -> np.ndarray:
    # image = cv2.resize(image, (520, 114), cv2.INTER_AREA)
    transformed = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

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
    contours_img = cv2.drawContours(image, [inner_contour], 0, (0, 0, 255), 2)

    # contours_img = cv2.drawContours(image, contours[:15], -1, (0, 220, 0), 1)
    # cv2.imshow('cntrs', contours_img)

    dst = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    dst = cv2.drawContours(dst, [inner_contour], 0, (0, 0, 255), 2)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    

    image[dst>0.01*dst.max()]=[0,255,0]
    # cv2.imshow('harris', image)

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
    # cv2.imshow('harris', image)

    
    best_matches = np.float32(best_matches)
    corners = corners[:, [1, 0]]
    best_matches = best_matches[:, [1, 0]]
    matrix = cv2.getPerspectiveTransform(best_matches, corners)
    transformed = cv2.warpPerspective(transformed, matrix, (y_end, x_end))

    cv2.imshow('transformed', transformed)

    return transformed

def extract_letters(plate: np.ndarray):
    plate = cv2.resize(plate, (1040, 228), cv2.INTER_AREA)

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

    # cv2.imshow('gray', padded_image)
    # padded_image = cv2.bitwise_not(padded_image)
    contours, hierarchy = cv2.findContours(padded_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    dst = np.zeros((padded_image.shape[0], padded_image.shape[1], 3), dtype=np.uint8)

    # print('cntr')
    # for controur in contours:
    #     print(cv2.contourArea(controur))
    #     if cv2.contourArea(controur) < 50_000:
    largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    
    contours = list(contours)
    del contours[largest_contour_index]
    
    print('height:' ,dst.shape[0])
    print('next')
    # dst = cv2.drawContours(dst, contours, -1, (255, 0, 0), 2)
    letters =[]
    for idx, contour in enumerate(contours):
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # print('cX: ', cX, ', cY: ', cY)
        # put text and highlight the center
        distX = 50
        distY = 50
        area = cv2.contourArea(contour)

        # print('cnt: ', contour)
        if area < 1000 or area > 25000:
            # del contours[idx]
            print('deleted')
            
            continue
        
        if cX < distX or cX > dst.shape[1] - distX or cY < distY or cY > dst.shape[0] - distY:
            continue
        # cv2.circle(dst, (cX, cY), 5, (255, 255, 255), -1)
        # cv2.circle(dst, (cX, cY), 5, (255, 255, 255), -1)
        letters.append(idx)
    contours_filtered = [val for idx, val in enumerate(contours) if idx in letters ]
    # print(contours_filtered)
    # print('letter len: ', len(letters))
    # print('letters: ', letters)


    cv2.fillPoly(dst, pts = contours_filtered, color=(255,255,255))

    for cnt in contours_filtered:
        if cv2.contourArea(cnt) < 3500:
            cv2.fillPoly(dst, pts = [cnt], color=(0,0,0))
            print('filled artefacts')





    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # dst = cv2.bitwise_not(dst)
    cv2.imshow('letters', dst)


    # 
    # canvas.fill(255)
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    contours_filtered = []
    for i, contour in enumerate(contours):
        # Retrieve the hierarchy information for the current contour
        hierarchy_info = hierarchy[0][i]

        # Check if the contour is an external contour (no parent and no child)
        if hierarchy_info[3] == -1:
            contours_filtered.append(contour)

    contours = contours_filtered
    contours_img = cv2.drawContours(dst, contours, -1, (255,255,255), 2)

    dst = cv2.bitwise_not(dst)
    cv2.imshow('inverted', contours_img)

    # contours to polygons - bounding boxes
    # contours_poly = [None]*len(contours)
    # boundRect = [None]*len(contours)
    # centers = [None]*len(contours)
    # radius = [None]*len(contours)
    boundRect =[]
    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True)
        rect = cv2.boundingRect(contours_poly)
        # x, y, w, h = rect 
        # area = w*h
        # print('area: ', area)
        # if area > 1000:
        boundRect.append(rect)
    # for rect in boundRect:
        
    #     print('width: ', w, ', height: ', h)
        # centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    
    # canvas = np.zeros((dst.shape[0], dst.shape[1], 3), dtype=np.uint8)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    boundRect = sorted(boundRect, key = lambda box: box[0])
    for i in range(len(boundRect)):
        # cv2.rectangle(dst, (int(boundRect[i][0])-10, int(boundRect[i][1])-10), \
        #   (int(boundRect[i][0]+boundRect[i][2])+10, int(boundRect[i][1]+boundRect[i][3])+10), (0,0,255), 2)

        # cv2.imshow('boxes', dst)

        x, y, w, h = boundRect[i]
        letter = dst[max(y-5, 0):min(y+h+5, dst.shape[0]), max(x-5, 0):min(x+w+5, dst.shape[1])]
        # cv2.imshow('letter', letter)
        # cv2.waitKey(100)
        
        
    

def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    
    cv2.namedWindow('image')
    # creating trackbars for red color change
    cv2.createTrackbar('blur_main', 'image', 3, 20, nothing)

    cv2.createTrackbar('blur1', 'image', 3, 50, nothing)
    cv2.createTrackbar('blur2', 'image', 10, 50, nothing)

    cv2.createTrackbar('brightness', 'image', 5, 255, nothing)
    cv2.createTrackbar('contrast', 'image', 20, 130, nothing)

    cv2.createTrackbar('sigma0', 'image', 35, 100, nothing)
    cv2.createTrackbar('sigma1', 'image', 1, 100, nothing)
    cv2.createTrackbar('sigma2', 'image', 11 , 100, nothing)
    
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


        # TODO: TRY TO ADJUST LATER, AFTER BLUR

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

    
        # gray = cv2.bilateralFilter(gray, 10, 20, 20, cv2.BORDER_DEFAULT)
        img_blur = cv2.GaussianBlur(gray, (gw, gw), gs)
        g1 = cv2.GaussianBlur(img_blur, (gw1, gw1), gs1)
        g2 = cv2.GaussianBlur(img_blur, (gw2, gw2), gs2)
        ret, thg = cv2.threshold(g2-g1, 160, 255, cv2.THRESH_OTSU)


        # cv2.imshow('canny', thg)
        # print('blur main c: ', blur_main_c)
        # open = cv2.erode(thg, (blur_main_c, blur_main_c))
        # cv2.imshow('open + close ', open)

        contours, hier = cv2.findContours(thg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        contours_img = np.zeros((dsty, dstx, 3), dtype=np.uint8)

        contours_img = cv2.drawContours(contours_img, contours, -1, (220, 220, 220), 1)
        # cv2.imshow('countours', contours_img)

        for i in range(len(contours)):
            if hier[0][i][2] == -1:
                continue
            
            x ,y, w, h = cv2.boundingRect(contours[i])
            a=w*h    
            aspectRatio = float(w)/h
            if  aspectRatio >= 2 and a > 25000:          
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

                    ex = 5
                    plate1 = img_cpy[ start_y-ex:end_y+ex, start_x-ex:end_x+ex].copy()
                    plate1 = img_cpy[ start_y:end_y, start_x:end_x].copy()
                    plate2 = thg[ start_y:end_y, start_x:end_x].copy()
                    # cv2.imshow('plate', plate2)
                    # extract_letters(cv2.cvtColor(plate1, cv2.COLOR_BGR2GRAY))
                    # find_strongest_line(plate1)

                    # contours_img = contours_img[ start_y-20:end_y+20, start_x-20:end_x+20].copy()
                    transformed = transform_corners(plate1)
                    extract_letters(transformed)

                    cv2.rectangle(img_cpy, (start_x,start_y), (end_x,end_y), color, 3)
                    cv2.putText(img_cpy, "rectangle "+str(x)+" , " + str(y-5), (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    
                    break
        cv2.imshow('image', img_cpy)


        key = cv2.waitKey(10)
        if key == ord('q'):
            cv2.destroyAllWindows()
            quit()
        if key == ord(' '):
            return 'PO12345'

        # cv2.waitKey(100)
        return 't'

