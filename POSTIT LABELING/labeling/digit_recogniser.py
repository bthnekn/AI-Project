from re import T
import numpy as np
import cv2
import mnist_test
kernel = np.ones((5,5),np.uint8)



def get_img_contour_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,3)
    thresh1 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,7,3.5)
    #thresh1 = cv2.erode(thresh1,kernel,iterations = 1)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    thresh1 = (255-thresh1)
    return img, contours, thresh1
loaded_model = mnist_test.model()
def main(img):

        img, contours, thresh = get_img_contour_thresh(img)
        ans1 = ''

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                x, y, w, h = cv2.boundingRect(contour)
                newImage = thresh[y:y + h, x:x + w]
                newImage = cv2.resize(newImage, (28, 28))
                newImage = np.array(newImage)
                newImage = newImage.flatten()
                newImage = newImage.reshape(1, 1,28,28)
                ans1 = loaded_model.predict(newImage)
                ans1=ans1.tolist()
                ans1 = ans1[0].index(max(ans1[0]))

        x, y, w, h = 0, 0, 300, 300
        return str(ans1)