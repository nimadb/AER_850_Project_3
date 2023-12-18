import cv2
import numpy as np

# Load the image
image = cv2.imread('/Users/nima.db/Documents/Fall 2023/AER850/Project 3/AER_850_Project_3/motherboard_image.JPEG')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise removal using morphological operations (erosion and dilation)
kernel = np.ones((10, 10), np.uint8)
cleaned_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

# # Perform adaptive thresholding
# adaptive_thresholded = cv2.adaptiveThreshold(
#     cleaned_image,
#     maxValue=255,
#     adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     thresholdType=cv2.THRESH_BINARY,
#     blockSize=31,
#     C=2
# )

_, adaptive_thresholded = cv2.threshold(cleaned_image, 140, 255, cv2.THRESH_TOZERO)


# Find contours
contours, _ = cv2.findContours(adaptive_thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

threshold_min = 10000  # Adjust this value as needed
threshold_max = 10000000  # Adjust this value as needed

filtered_contours = [cnt for cnt in contours if threshold_min < cv2.contourArea(cnt) < threshold_max]

# Draw contours on the original image
contour_image = image.copy()
cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 3)  # Draw all contours in green with thickness 3

# Create a mask using the filtered contours
mask = np.zeros_like(gray_image)
cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

# Morphological closing to help close gaps in the mask
closing_kernel = np.ones((1520, 1520), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)

# Dilation to further fill in and expand the contours
dilation_kernel = np.ones((10, 10), np.uint8)
mask = cv2.dilate(mask, dilation_kernel, iterations=1)

# Find the outermost contour
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
outer_contour = contours[0]  # Assuming the first contour is the outermost

# Draw the outer contour to enclose the filled areas
cv2.drawContours(mask, [outer_contour], -1, 255, thickness=cv2.FILLED)

# Apply the mask on the original image
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Gray Scale Image', gray_image)

cv2.imshow('cleaned_image', cleaned_image)
cv2.imshow('adaptive_thresholded', adaptive_thresholded)

# # cv2.imshow('edges', edges)
cv2.imshow('Image with Contours', contour_image)
cv2.imshow('mask', mask)
cv2.imshow('Masked Image', masked_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('gray_image.png', gray_image)
cv2.imwrite('cleaned_image.png', cleaned_image)
cv2.imwrite('adaptive_thresholded.png', adaptive_thresholded)
cv2.imwrite('Image_with_Contours.png', contour_image)
cv2.imwrite('mask.png', mask)
cv2.imwrite('extracted_motherboard.png', masked_image)