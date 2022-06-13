import cv2 as cv
import numpy as np
import imutils   #image rotation
#using the rotate_image function for rotation
def rotate_image(image, angle):     
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    #the centre is located at h/2,w/2
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    #image matrix rotated
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    #result stores the modified image rotated.
    return result
'''dimensions remain unchanged.'''

img=cv.imread('CVtask.jpg')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray, 225, 255, cv.THRESH_BINARY)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
n1=0
n2=0
n3=0
n4=0
'''in the hence if statements,we want to prevent the repitition of if statements.'''

'''contour detection'''
top = np.zeros((img.shape[0],img.shape[1],3),np.uint8)*255

for c in contours: 
    approx = cv.approxPolyDP(c, 0.01 * cv.arcLength(c,True), True)
if (len(approx)==4 and cv.contourArea(c)>50):#using contours to remove extra noise elements in the image.
    if (len(approx)==4):
        rect=cv.minAreaRect(c)
        '''rect stores the coordinates of the image with leaast area among the actual h-w area and contour detected area.'''
       
        t=rect[1][1]/rect[1][0]
        if (t>0.9 and t<1.1):
            
            x=int(rect[0][0])
            y=int(rect[0][1])
            #all squares are detected
          
            if(img[y,x][0]==79 and img[y,x][1]==209 and img[y,x][2]==146 and n1==0):
                '''img[y,x]returns bgr code at the pixel on comparing color'''
                n1=n1+1
                wt = cv.imread('LMAO.jpg')
                '''id of aruco markers found'''
                gray = cv.cvtColor(wt, cv.COLOR_BGR2GRAY)
                edged = cv.Canny(gray, 30, 200)
                cc, hh = cv.findContours(edged,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                #contours
                
                '''align the aruco marker with horizontal or vertical'''
               
                rect1=cv.minAreaRect(cc[0])#only one element in cc
                #minarearect function returns angle at index 2
                w1=rotate_image(wt, rect1[2])
                area1=rect1[1][1]*rect1[1][0]
                '''storing area(explained later why)'''
                
                gray1 = cv.cvtColor(w1, cv.COLOR_BGR2GRAY)
                edged1 = cv.Canny(gray1, 30, 200)
                cc1, hh1 = cv.findContours(edged1,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                '''finding contours of aligned aruco marker'''
                l=[]
                for k in range(len(cc1)):
                    l.append(abs(cv.contourArea(cc1[k])-area1))
                '''finding the contour which fits the aruco marker best based on the minimum difference between the actual area
                contour detected area.'''
                
                inde=l.index(min(l))
                k1=cc1[inde]
                #storing the contour for further use
                
                xoff1,yoff1,w1,h1 = cv.boundingRect(k1)
                x_end1 = int(xoff1 + w1)
                y_end1= int(yoff1 + h1)
                w1=w1[yoff1:y_end1,xoff1:x_end1]
                #cropping the aligned aruco marker
                
                w1 = cv.resize(w1,(int(rect[1][0]),int(rect[1][1])))
                #rotating aruco marker for putting the main image and blank image
                tran = np.ones((int(rect[1][0]),int(rect[1][1]),3),np.uint8)*255
                #removing black area in the corners
                
                tran=imutils.rotate_bound(tran, rect[2])
                #rotating image  for puting on main image 
                
                
                tran2 = np.zeros((int(rect[1][0]),int(rect[1][1]),3),np.uint8)*255
                tran2=imutils.rotate_bound(tran2, rect[2])
                tran=cv.bitwise_not(tran)
                w1=imutils.rotate_bound(w1, rect[2])
                
                wt2=cv.bitwise_or(tran2,w1)
                w1=cv.bitwise_or(tran,w1)
                xoff=int(rect[0][0] - w1.shape[0]/2)
                yoff=int(rect[0][1] - w1.shape[0]/2)
                x_end = int(rect[0][0] + w1.shape[1]/2)
                y_end = int(rect[0][1] + w1.shape[0]/2)
                new=img[yoff:y_end,xoff:x_end]
                # here we finally place the aruco markers on main image and image top
                temp=cv.bitwise_not(w1)
                temp2=cv.bitwise_not(new)
                new=cv.add(temp,temp2)
                new=cv.bitwise_not(new)
                top[yoff:y_end,xoff:x_end] = wt2
                img[yoff:y_end,xoff:x_end] = new
                
                
                
                '''repeated if statements for the subsequent aruco markers'''
            if(img[y,x][0]==9 and img[y,x][1]==127 and img[y,x][2]==240 and n2==0):
                n2=n2+1
                xoff,yoff,w,h = cv.boundingRect(c)
                wt = cv.imread('XD.jpg')
                gray = cv.cvtColor(wt, cv.COLOR_BGR2GRAY)
                edged = cv.Canny(gray, 30, 200)
                cc, hh = cv.findContours(edged,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
               
   
                rect1=cv.minAreaRect(cc[0])
         
                w1=rotate_image(wt, rect1[2])
                area1=rect1[1][1]*rect1[1][0]
                
                gray1 = cv.cvtColor(w1, cv.COLOR_BGR2GRAY)
                edged1 = cv.Canny(gray1, 30, 200)
                cc1, hh1 = cv.findContours(edged1,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                l=[]
                for k in range(len(cc1)):
                    l.append(abs(cv.contourArea(cc1[k])-area1))
                inde=l.index(min(l))
                k1=cc1[inde]
                xoff1,yoff1,w1,h1 = cv.boundingRect(k1)
                x_end1 = int(xoff1 + w1)
                y_end1= int(yoff1 + h1)
                w1=w1[yoff1:y_end1,xoff1:x_end1]
                
                w1 = cv.resize(w1,(int(rect[1][0]),int(rect[1][1])))
                tran = np.ones((int(rect[1][0]),int(rect[1][1]),3),np.uint8)*255
                tran=imutils.rotate_bound(tran, rect[2])
                tran2 = np.zeros((int(rect[1][0]),int(rect[1][1]),3),np.uint8)*255
                tran2=imutils.rotate_bound(tran2, rect[2])
                tran=cv.bitwise_not(tran)
                w1=imutils.rotate_bound(w1, rect[2])
                               
                wt2=cv.bitwise_or(tran2,w1)
                w1=cv.bitwise_or(tran,w1)
                xoff=int(rect[0][0] - w1.shape[0]/2)
                yoff=int(rect[0][1] - w1.shape[0]/2)
                x_end = int(rect[0][0] + w1.shape[1]/2)
                y_end = int(rect[0][1] + w1.shape[0]/2)
                new=img[yoff:y_end,xoff:x_end]
                temp=cv.bitwise_not(w1)
                temp2=cv.bitwise_not(new)
                new=cv.add(temp,temp2)
                new=cv.bitwise_not(new)
                top[yoff:y_end,xoff:x_end] = wt2
                img[yoff:y_end,xoff:x_end] = new


            if(img[y,x][0]==210 and img[y,x][1]==222 and img[y,x][2]==228 and n3==0):
                n3=n3+1
                xoff,yoff,w,h = cv.boundingRect(c)
                wt = cv.imread('HaHa.jpg')
                gray = cv.cvtColor(wt, cv.COLOR_BGR2GRAY)
                edged = cv.Canny(gray, 30, 200)
                cc, hh = cv.findContours(edged,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
               
   
                rect1=cv.minAreaRect(cc[0])
         
                w1=rotate_image(wt, rect1[2])
                area1=rect1[1][1]*rect1[1][0]
                
                gray1 = cv.cvtColor(w1, cv.COLOR_BGR2GRAY)
                edged1 = cv.Canny(gray1, 30, 200)
                cc1, hh1 = cv.findContours(edged1,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                l=[]
                for k in range(len(cc1)):
                    l.append(abs(cv.contourArea(cc1[k])-area1))
                inde=l.index(min(l))
                k1=cc1[inde]
                xoff1,yoff1,w1,h1 = cv.boundingRect(k1)
                x_end1 = int(xoff1 + w1)
                y_end1= int(yoff1 + h1)
                w1=w1[yoff1:y_end1,xoff1:x_end1]
                
                w1 = cv.resize(w1,(int(rect[1][0]),int(rect[1][1])))
                tran = np.ones((int(rect[1][0]),int(rect[1][1]),3),np.uint8)*255
                tran=imutils.rotate_bound(tran, rect[2])
                tran2 = np.zeros((int(rect[1][0]),int(rect[1][1]),3),np.uint8)*255
                tran2=imutils.rotate_bound(tran2, rect[2])
                tran=cv.bitwise_not(tran)
                w1=imutils.rotate_bound(w1, rect[2])
                               
                wt2=cv.bitwise_or(tran2,w1)
                w1=cv.bitwise_or(tran,w1)
                xoff=int(rect[0][0] - w1.shape[0]/2)
                yoff=int(rect[0][1] - w1.shape[0]/2)
                x_end = int(rect[0][0] + w1.shape[1]/2)
                y_end = int(rect[0][1] + w1.shape[0]/2)
                new=img[yoff:y_end,xoff:x_end]
                temp=cv.bitwise_not(w1)
                temp2=cv.bitwise_not(new)
                new=cv.add(temp,temp2)
                new=cv.bitwise_not(new)
                top[yoff:y_end,xoff:x_end] = wt2
                img[yoff:y_end,xoff:x_end] = new


            if(img[y,x][0]==0 and img[y,x][1]==0 and img[y,x][2]==0 and n4==0):
                n4=n4+1
                xoff,yoff,w,h = cv.boundingRect(c)
                wt = cv.imread('Ha.jpg')
                gray = cv.cvtColor(wt, cv.COLOR_BGR2GRAY)
                edged = cv.Canny(gray, 30, 200)
                cc, hh = cv.findContours(edged,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
               
   
                rect1=cv.minAreaRect(cc[0])
                w1=rotate_image(wt, rect1[2])
                area1=rect1[1][1]*rect1[1][0]
                
                gray1 = cv.cvtColor(w1, cv.COLOR_BGR2GRAY)
                edged1 = cv.Canny(gray1, 30, 200)
                cc1, hh1 = cv.findContours(edged1,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                l=[]
                for k in range(len(cc1)):
                    l.append(abs(cv.contourArea(cc1[k])-area1))
                inde=l.index(min(l))
                k1=cc1[inde]
                xoff1,yoff1,w1,h1 = cv.boundingRect(k1)
                x_end1 = int(xoff1 + w1)
                y_end1= int(yoff1 + h1)
                w1=w1[yoff1:y_end1,xoff1:x_end1]
                
                w1 = cv.resize(w1,(int(rect[1][0]),int(rect[1][1])))
                tran = np.ones((int(rect[1][0]),int(rect[1][1]),3),np.uint8)*255
                tran=imutils.rotate_bound(tran, rect[2])
                tran2 = np.zeros((int(rect[1][0]),int(rect[1][1]),3),np.uint8)*255
                tran2=imutils.rotate_bound(tran2, rect[2])
                tran=cv.bitwise_not(tran)
                w1=imutils.rotate_bound(w1, rect[2])
              
                wt2=cv.bitwise_or(tran2,w1)
              
                w1=cv.bitwise_or(tran,w1)
                xoff=int(rect[0][0] - w1.shape[0]/2)
                yoff=int(rect[0][1] - w1.shape[0]/2)
                x_end = int(rect[0][0] + w1.shape[1]/2)
                y_end = int(rect[0][1] + w1.shape[0]/2)
              
                new=img[yoff:y_end,xoff:x_end]
                temp=cv.bitwise_not(w1)
                temp2=cv.bitwise_not(new)
                new=cv.add(temp,temp2)
                new=cv.bitwise_not(new)
                top[yoff:y_end,xoff:x_end] = wt2
                img[yoff:y_end,xoff:x_end] = new
               
img=cv.bitwise_or(img,top)
#As we know,aruco markers have white  colour  inside.the above function makes sure this happens.
img=cv.resize(img,(1000,500))
cv.imshow('sdfsdfabc',img)

cv.waitKey(0)