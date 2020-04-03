
import  cv2
import  numpy as np
import compare

def main():
    sample=cv2.imread("./data/monoEraser.jpg")
    testImg=cv2.imread("./data/test2d.jpg")
    height, width, channels = testImg.shape
    image_size = height * width
    testImg_b= cv2.GaussianBlur(testImg, (9, 9), 2)
    hsv=cv2.cvtColor(testImg_b,cv2.COLOR_BGR2HSV)
    lower = np.array([110, 50, 50])
    upper = np.array([240, 255, 255])
    frame_mask = cv2.inRange(hsv, lower, upper)

    img2,blueArea = getColorArea(frame_mask, testImg)
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 50])
    frame_mask2=cv2.inRange(hsv,lower,upper)

    kernel = np.ones((5, 5), dtype=np.uint8)

    frame_mask2 = cv2.dilate(frame_mask2, kernel)
    frame_mask2 = cv2.erode(frame_mask2, kernel)
    #frame_mask2 = cv2.erode(frame_mask2, kernel)
    #frame_mask2=cv2.dilate(frame_mask2,kernel)


    testImg_g=cv2.cvtColor(testImg,cv2.COLOR_BGR2GRAY)

    img_canny=cv2.Canny(frame_mask2,300,400)


    img,blackArea=getColorArea(frame_mask2, testImg)
    rects=[]
    for i, blue_a in enumerate(blueArea) :
        for j,black_a in enumerate(blackArea):
            b,rect=checkAreaRatio(testImg,blue_a,black_a)
            if(b):
                rects.append(rect)
    deleteRects=[]
    for i, rect1 in enumerate(rects):
        for j,rect2 in enumerate(rects):
            if(i>j):
                k= checkContain(rect1,rect2,height,width)
                if(k==1):
                    deleteRects.append(rect2)
                elif(k==2):
                    deleteRects.append(rect1)
    for rect1 in deleteRects:
        rects.remove(rect1)
    deleteRects=[]
    for rect1 in rects:
        if(not compareTrait(sample,img,rect1)):
            deleteRects.append(rect1)
    for rect1 in deleteRects:
        rects.remove(rect1)
    #debugRect(testImg,rects)
    for rect1 in rects:
        img=changeColorItaly(img,rect1)


    cv2.imshow("sample2", img)
    cv2.imwrite("./data/output10.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def changeColorItaly(img,rect):
    height, width, channels = img.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    mask1 = np.copy(mask)
    mask1 = cv2.drawContours(mask1, [box], 0, 255, 1)
    mask1 = fillArea(mask1, int(rect[0][0] + 2), int(rect[0][1] + 2))
    mask2 = np.copy(mask)
    mask2 = cv2.drawContours(mask2, [box], 0, 255, 1)
    mask2 = fillArea(mask2, int(rect[0][0] + 2), int(rect[0][1] + 2))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([110, 50, 50])
    upper = np.array([240, 255, 255])
    frame_mask = cv2.inRange(hsv, lower, upper)
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 50])
    frame_mask2 = cv2.inRange(hsv, lower, upper)
    mask1=cv2.bitwise_and(frame_mask,mask1)
    mask2=cv2.bitwise_and(frame_mask2,mask2)
    mask1=cv2.cvtColor(mask1,cv2.COLOR_GRAY2BGR)
    mask2=cv2.cvtColor(mask2,cv2.COLOR_GRAY2BGR)
    for x in range(height):
        for y in range(width):
            b, g, r = mask1[x, y]
            if (b, g, r) == (0, 0, 0):
                continue
            img[x, y] = 99, 135, 0
    for x in range(height):
        for y in range(width):
            b, g, r = mask2[x, y]
            if (b, g, r) == (0, 0, 0):
                continue
            img[x, y] = 57, 41, 206
    return img

def compareTrait(sample,img,rect):
    height, width, channels = img.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    mask1 = np.copy(mask)
    mask1 = cv2.drawContours(mask1, [box], 0, 255, 1)
    mask1=fillArea(mask1,int(rect[0][0]+2),int(rect[0][1]+2))

    mask1=cv2.cvtColor(mask1,cv2.COLOR_GRAY2BGR)
    img2=cv2.bitwise_and(img,mask1)

    cv2.imshow("sample3", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return compare.compare(sample,img2)
def debugRect(img,rects):
    for rect in rects:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img2 = np.copy(img)
        img2 = cv2.drawContours(img2, [box], 0,255, 1)
        cv2.imshow("sdf", img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def checkContain(rect1,rect2,height,width):#領域の中に領域があるかチェック
    mask = np.zeros((height , width ), dtype=np.uint8)
    box=cv2.boxPoints(rect1)
    box = np.int0(box)
    mask1 = np.copy(mask)
    mask1 = cv2.drawContours(mask1, [box], 0, 255, 1)
    mask1=fillArea(mask1,int(rect1[0][0]+2),int(rect1[0][1]+2))
    box = cv2.boxPoints(rect2)
    box = np.int0(box)
    mask2 = np.copy(mask)
    mask2 = cv2.drawContours(mask2, [box], 0, 255, 1)
    mask2 = fillArea(mask2, int(rect2[0][0] + rect2[1][0] / 2),int( rect2[0][1] + rect2[1][1] / 2))
    mask3=cv2.bitwise_and(mask1,mask2)
    contours, hierarchy = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if(len(contours)<=0):
        return  0
    area3 = cv2.contourArea(contours[0])

    area1=rect1[1][0]*rect1[1][1]
    area2 = rect2[1][0] * rect2[1][1]
    ratio=0.9
    if(area3>area1*ratio):
        return  1
    elif(area3>area2*ratio):
        return  2
    else:
        return 0

def checkAreaRatio(img,mask1,mask2):
    mask1,cnt_mask1=mask1
    mask2, cnt_mask2 = mask2
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask4=cv2.dilate(mask1,kernel)
    mask5=cv2.dilate(mask2,kernel)
    mask6=cv2.bitwise_and(mask4,mask5)

    #cv2.imwrite("./data/output7.png", mask1)
    #cv2.imwrite("./data/output8.png", mask2)
    #temp=cv2.bitwise_or(mask4,mask5)


    if(np.average(mask6)>0):#隣り合っている領域を除外
        return False,None
    area1=cv2.contourArea(cnt_mask1)
    area2=cv2.contourArea(cnt_mask2)
    cnt_mask3=np.concatenate([cnt_mask1,cnt_mask2])

    rect = cv2.minAreaRect(cnt_mask3)
    area3= rect[1][0]*rect[1][1]

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img2 = np.copy(img)
    img2 = cv2.drawContours(img2, [box], 0, (0, 0, 255), 2)

    #temp = cv2.drawContours(cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR), [box], 0, (0, 0, 255), 2)
    #cv2.imwrite("./data/output9.png", temp)

    if (area3 > (area1 + area2) * 3 or area3<(area1 + area2) * 1.6):#比率で除外
        return False,None

    return  True,rect

def getColorArea(frame_mask, draw_img, draw=False):
    ret, testImg_g2 = cv2.threshold(frame_mask, 100, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(testImg_g2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width, channels = draw_img.shape

    img=draw_img
    if(draw):
        img = cv2.drawContours(draw_img, contours, -1, (0, 0, 255, 255), 2, cv2.LINE_AA)
    cont_area=[]

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        image_size=height*width
        if area < 500:
            continue
        if image_size * 0.99 < area:
            continue
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if(draw):
            cv2.drawContours(img,[approx], -1,  (255, 0, 255), 2)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [approx], -1, (255, 0, 255), 2)
        M = cv2.moments(contour)
        mask= fillArea(mask,int(M['m10']/M['m00']),int(M['m01']/M['m00']))#引数に重心を入れている

        #cv2.imshow("aa",mask)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        cont_area.append((mask,contour))




    return  img,cont_area
def fillArea(img,startx,starty,color=(0,0,255)):
    channels=0
    height=0
    width=0
    if(img.ndim==2):
        channels=1
        height, width = img.shape
    elif(img.ndim==3):
        height, width, channels = img.shape
    mask = np.zeros((height+2, width+2), dtype=np.uint8)#+2しないとエラー
    if(channels==3):
        pass
    else:
        color=255
    retval, img2, mask, rect = cv2.floodFill(img, mask, seedPoint=(startx, starty), newVal=color)
    if(np.average(img2)>100):#もし塗りつぶしが多ければ
        img2=cv2.bitwise_not(img2)#反転
    return img2

if __name__ == '__main__':
    main()