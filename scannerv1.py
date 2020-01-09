#! /usr/bin/env python3

# Author: Daniel Mallia

# This short program should perform the basic function of a document scanner "app":
# 1. Retrieve an image
# 2. Perform edge detection
# 3. Perform a perspective transformation for a better image for OCR
# 4. Sharpen the image.
# 5. Save the image.

import cv2 as cv
import sys
import copy

def sortByArea(contour):
    return cv.contourArea(contour)

def transform(document):
    documentCopy = copy.deepcopy(document)
    documentCopy = cv.cvtColor(documentCopy, cv.COLOR_RGB2GRAY)
    _, documentCopy = cv.threshold(documentCopy, 180, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(documentCopy, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours.sort(key = sortByArea)
    #cv.drawContours(document, contours, (len(contours) - 1), (0, 0, 255), 10)
    possibleDocument = contours[-1]
    
    epsilon = 0.1*cv.arcLength(possibleDocument,True)
    approx = cv.approxPolyDP(possibleDocument,epsilon,True)

    return document

if __name__ == "__main__":
    # Usage:
    if(len(sys.argv) != 2):
        print('Usage: ./scannerv1.py [IMAGE]')
        sys.exit()

    imageName = sys.argv[1]
    document = cv.imread(imageName, cv.IMREAD_COLOR)
    
    document = transform(document)

    cv.namedWindow('final', cv.WINDOW_NORMAL)
    cv.resizeWindow('final', 800, 600)
    
    # Display
    cv.imshow('final', document)
    cv.waitKey(0)
    cv.destroyWindow('final')

    # Save the image
