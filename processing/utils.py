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


def find_strongest_line(image):
    h, w = image.shape[:2]
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 70, 120, apertureSize=3)

    # Perform Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    if lines is not None:
        # Find the strongest line
        lines = sorted(lines, key=lambda line: line[0][1])
        strongest_line = lines[0]

        # Get the parameters of the strongest line
        rho = strongest_line[0][0]
        theta = strongest_line[0][1]

        # Calculate the line's coordinates
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))  
        y2 = int(y0 - 1000 * (a))


        # corner points for perspective transform:
        # top/bottom left/right
        xtl = 0
        xbl = 0
        xtr = w
        print('rho: ', rho)
        print('theta: ', theta)
        print('w: ', w, 'h: ', h)
        # ytl = int(rho / a)
        # ybl = int( (y0 - w * b) / a)

        ytr = int( (y0 - (w * b)) / a)
        print('height: ', h, ', y: ', ytr)
        cv2.circle(image, (0, int(rho)), 3, (0, 255, 0), 5)
        cv2.circle(image, (int(w), 180), 3, (0, 255, 0), 5)

        cv2.imshow('image', image)
        # Draw the strongest line on the image
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Compute the transformation matrix
        angle_deg = np.degrees(theta) - 90
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        # rest :
        for line in lines:
            # Get the parameters of the strongest line
            rho = line[0][0]
            theta = line[0][1]

            # Calculate the line's coordinates
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Draw the strongest line on the image
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Perform the image transformation
        transformed_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))


        cv2.imshow('transformed_image', transformed_image)

    cv2.imshow('strongest line', image)

def extract_letters(plate):
    h, w = plate.shape[:2]

    ret, plate = cv2.threshold(plate,50,255,cv2.THRESH_BINARY)
    # kernel = np.ones((3,3),np.uint8)
    # plate = cv2.dilate(plate,kernel,iterations = 1)
    # Determine contour of all blobs found
    contours, hierarchy = cv2.findContours( plate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours]

    # Draw all contours
    vis = np.zeros((h, w, 3), np.uint8)
    cv2.drawContours( vis, contours, -1, (128,255,255), 3, cv2.LINE_AA)
    cv2.imshow('vis1', vis)
    # Draw the contour with maximum perimeter (omitting the first contour which is outer boundary of image
    # Not necessary in this case
    vis2 = np.zeros((h, w, 3), np.uint8)
    perimeter=[]
    for cnt in contours[1:]:
        perimeter.append(cv2.arcLength(cnt,True))
    # print perimeter
    # print max(perimeter)
    maxindex = perimeter.index(max(perimeter))
    # print maxindex

    cv2.drawContours( vis2, contours, maxindex +1, (255,0,0), -1)
    cv2.imshow('extract', vis2)


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


        cv2.imshow('canny', thg)
        # print('blur main c: ', blur_main_c)
        # open = cv2.erode(thg, (blur_main_c, blur_main_c))
        # cv2.imshow('open + close ', open)

        contours, hier = cv2.findContours(thg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

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

                    plate1 = img_cpy[ start_y-20:end_y+20, start_x-20:end_x+20].copy()
                    plate2 = thg[ start_y:end_y, start_x:end_x].copy()
                    cv2.imshow('plate', plate2)
                    # extract_letters(cv2.cvtColor(plate1, cv2.COLOR_BGR2GRAY))
                    find_strongest_line(plate1)
                    

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

