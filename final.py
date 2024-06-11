import cv2
import numpy as np

def region_of_interest(img, vertices):
  
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
  
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def draw_lines(img, lines, direction_label):
   
    if lines is None:
        return
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, direction_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def determine_direction(lines, img_width):
    
    if lines is None or len(lines) == 0:
        return "NO lines" 
    
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
   
        if slope < -0.5:
            right_lines.append(line)
        elif slope > 0.5:
            left_lines.append(line)
    
    if len(left_lines) > 0 and len(right_lines) > 0:
        return "Forward"
    elif len(left_lines) > 0:
        return "Left Turn"
    elif len(right_lines) > 0:
        return "Right Turn"
    
    return "no lines" 

def birdseye_view(frame):
  
    width, height = 1920, 1080

    src = np.float32([
        [0, 1080],
        [1920, 1080],
        [1632, 648],
        [288, 648]
    ])
      
    dst = np.float32([
        [0, height],         
        [width, height], 
        [width, 0],            
        [0, 0]
    ])
    
    matrix = cv2.getPerspectiveTransform(src, dst)
    
    warped = cv2.warpPerspective(frame, matrix, (width, height))
    
    return warped

def process_frame(frame):
    """Process a frame to detect lanes and determine direction."""
    height, width = frame.shape[:2]
    
    birdseye = birdseye_view(frame)
    
    grey = cv2.cvtColor(birdseye, cv2.COLOR_BGR2GRAY)
    

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    

    edges = cv2.Canny(blur, 50, 150)
    
    #ROI shape
    squareHeight = int(height * 0.4)
    squareWidth = int(width * 0.8)
    vertices = np.array([[
        (int((width - squareWidth) / 2), height),
        (int((width + squareWidth) / 2), height),
        (int((width + squareWidth) / 2), height - squareHeight),
        (int((width - squareWidth) / 2), height - squareHeight)
    ]], dtype=np.int32)
    
    cv2.polylines(birdseye, [vertices], isClosed=True, color=(0, 255, 255), thickness=2)
     
    roi = region_of_interest(edges, vertices)
    
    lines = hough_lines(roi, 1, np.pi / 180, 15, 10, 20)  
    

    direction = determine_direction(lines, width)
    print(f"Determined direction: {direction}")
    
    draw_lines(birdseye, lines, direction)
    
    return birdseye

if __name__ == "__main__":

    cap = cv2.VideoCapture('/path/to/video/IMG_0897.MOV')

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame)

        cv2.imshow('Lane Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
    

    cap.release()
    out.release()
    cv2.destroyAllWindows()
