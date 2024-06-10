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
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, direction_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def determine_direction(lines, img_width):
    """Determine the direction based on the slopes of the lines."""
    if lines is None or len(lines) == 0:
        return "Forward"  # Default to "Forward" if no lines detected
    
    left_lines = []
    right_lines = []
    left_slopes = []
    right_slopes = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue  # Skip vertical lines (slope calculation will fail)
        slope = (y2 - y1) / (x2 - x1)
        if slope > 0.5 and x1 < img_width / 2 and x2 < img_width / 2:
            left_lines.append(line)
            left_slopes.append(slope)
        elif slope < -0.5 and x1 > img_width / 2 and x2 > img_width / 2:
            right_lines.append(line)
            right_slopes.append(slope)
    
    if len(left_lines) > 0 and len(right_lines) > 0:
        return "Forward"
    elif len(left_lines) > 0:
        avg_left_slope = np.mean(left_slopes)
        print(f"Avg left slope: {avg_left_slope}")
        if avg_left_slope > 0.5:  # Adjust the threshold for better sensitivity
            return "Left Turn"
        else:
            return "Forward"  # Default to "Forward" if no clear direction
    elif len(right_lines) > 0:
        avg_right_slope = np.mean(right_slopes)
        print(f"Avg right slope: {avg_right_slope}")
        if avg_right_slope < -0.5:  # Adjust the threshold for better sensitivity
            return "Right Turn"
        else:
            return "Forward"  # Default to "Forward" if no clear direction
    
    return "Forward"  # Default to "Forward" if no clear direction

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
    lines = hough_lines(roi, 1, np.pi / 180, 15, 40, 20)
    
    # Debug: Print detected lines
    if lines is not None:
        print(f"Detected lines: {len(lines)}")
        for line in lines:
            x1, y1, x2, y2 = line[0]
            print(f"({x1}, {y1}) - ({x2}, {y2})")
    
    # Step 6: Determine Direction
    direction = determine_direction(lines, width)
    print(f"Determined direction: {direction}")
    
    # Step 7: Draw Lines and Label
    draw_lines(img, lines, direction)
    
    # Step 8: Display the Result
    cv2.imshow('Lane Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_image('/Users/charliefraser/Downloads/forwdd.jpg')
