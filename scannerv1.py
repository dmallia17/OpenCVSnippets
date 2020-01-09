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
import numpy as np

# Helper function - returns area for list sorting
def sortByArea(contour):
    return cv.contourArea(contour)

# Transformation function - "snaps" document to frame.
def transform(document):
    rows, cols, _ = document.shape
    max_row = rows - 1
    max_col = cols - 1
    documentCopy = copy.deepcopy(document)
    documentCopy = cv.cvtColor(documentCopy, cv.COLOR_RGB2GRAY) # Convert to grayscale

    # Apply Otsu binary thresholding
    _, documentCopy = cv.threshold(documentCopy, 180, 255, cv.THRESH_BINARY+cv.THRESH_OTSU) 

    # Find contours and select the contour which covers the most area - should be the desired document
    contours, hierarchy = cv.findContours(documentCopy, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours.sort(key = sortByArea, reverse=True) 
    #cv.drawContours(document, contours, 0, (0, 0, 255), 10) # Draw greatest area contour for debugging
    possibleDocument = contours[0]

    epsilon = 0.1*cv.arcLength(possibleDocument,True)
    approx = cv.approxPolyDP(possibleDocument,epsilon,True) # approximate contour with rectangle

    dest = np.float32([[0,0],[max_row,0], [0,max_col], [max_row, max_col]])
    # Fix the following hardcoding:
    matrix = cv.getPerspectiveTransform(np.float32([approx[0], approx[3], approx[1], approx[2]]), dest)
    
    # Apply perspective transformation
    transformedDocument = cv.warpPerspective(document, matrix, (max_row, max_col))

    return transformedDocument

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
