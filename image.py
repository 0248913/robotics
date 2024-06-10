import cv2
import numpy as np

def grayscale(img):
    """Convert the image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Apply the Canny edge detector."""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """Apply an image mask."""
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """Apply Hough Transform to detect lines."""
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def draw_lines(img, lines, direction_label):
    """Draw lines on the image and label them."""
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, direction_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def determine_direction(lines, img_width):
    """Determine the direction based on the position of the lines."""
    left_lines = []
    right_lines = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 < img_width / 2 and x2 < img_width / 2:
                left_lines.append(line)
            elif x1 > img_width / 2 and x2 > img_width / 2:
                right_lines.append(line)

    if left_lines and right_lines:
        return "Forward"
    elif left_lines:
        return "Right Turn"
    elif right_lines:
        return "Left Turn"
    else:
        return "Unknown"

def process_image(image_path):
    """Process an image to detect lanes and determine direction."""
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Step 1: Grayscale
    gray = grayscale(img)
    
    # Step 2: Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3: Canny Edge Detection
    edges = canny(blur, 50, 150)
    
    # Step 4: Region of Interest
    vertices = np.array([[(width // 4, height), (width // 4, 0), 
                          (3 * width // 4, 0), (3 * width // 4, height)]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)
    
    # Step 5: Hough Transform
    lines = hough_lines(roi, 1, np.pi / 180, 15, 100, 50)
    
    # Step 6: Determine Direction
    direction = determine_direction(lines, width)
    
    # Step 7: Draw Lines and Label
    draw_lines(img, lines, direction)
    
    # Step 8: Display the Result
    cv2.imshow('Lane Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_image('path_to_your_image.jpg')
