import numpy as np
import cv2

def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    

    resized = cv2.resize(image, (640, 400), cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.blur(gray, (9, 9))
    blur2 = cv2.blur(gray, (15, 15))

    # blur1 = cv2.bilateralFilter(gray, 11, 17, 17) 

    gray = blur1-blur2
    cv2.imshow('blur1 - blur2', gray)
    gray = cv2.bitwise_not(gray)
    edges = cv2.bilateralFilter(gray, 15, 80, 80, cv2.BORDER_DEFAULT) 
    # gray = cv2.medianBlur(gray, 5)
    # edges = cv2.Canny(gray, 80, 100, apertureSize=3)
   

    # get contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(resized, contours, -1, (0,255,0), 3)
    # cv2.waitKey()
    # sort contours
    # contours = sorted(contours, key=cv2.contourArea, reverse=True) [:30]
    
    screenCnt = None
    image2 = resized.copy()
    cv2.drawContours(image2, contours, -1, (0,255,0), 3)
    cv2.imshow("Top 30 contours",image2)
    # cv2.waitKey()

    # contour with 4 sides
    # i=7
    # for c in contours: 
    #     perimeter = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
    #     if len(approx) == 4: 
    #             screenCnt = approx

    # crop 
    # x,y,w,h = cv2.boundingRect(c) 
    # new_img=image[y:y+h,x:x+w]

    # cv2.imshow('plate', new_img)

    # cv2.drawContours(resized, screenCnt, -1, (255, 0, 0), 3)
    cv2.imshow("image with detected license plate", resized)


    cv2.imshow('edges', edges)
    cv2.imshow('t', resized)
    cv2.waitKey()
    

    return 'PO12345'
