import cv2
import numpy as np

def regionOF(img, vertices):

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def houghLines(img, rho, theta, threshold, min_line_len, max_line_gap):

    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def drawLines(img, lines, direction_label):

    if lines is None:
        return
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, direction_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def directionDET(lines, img_width):
 
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

def BEV(frame):

    width, height = 1920, 1080

    
    pst1 = np.float32([
        [0, 1080],
        [1920, 1080],
        [1632, 648],
        [288, 648]
    ])
    

    pst2 = np.float32([
        [0, height],         
        [width, height], 
        [width, 0],            
        [0, 0]
    ])
    
    
    M = cv2.getPerspectiveTransform(pst1, pst2)
    

    warped = cv2.warpPerspective(frame, M, (width, height))
    
    return warped

def processed(frame):

    height, width = frame.shape[:2]
    
 
    birdseye = BEV(frame)
    

    grey = cv2.cvtColor(birdseye, cv2.COLOR_BGR2GRAY)
    

    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    

    edges = cv2.Canny(blur, 50, 150)
    
    
    heightS = int(height * 0.4)
    widthS = int(width * 0.8)
    vertices = np.array([[
        (int((width - widthS) / 2), height),
        (int((width + widthS) / 2), height),
        (int((width + widthS) / 2), height - heightS),
        (int((width - widthS) / 2), height - heightS)
    ]], dtype=np.int32)
    

    cv2.polylines(birdseye, [vertices], isClosed=True, color=(0, 255, 255), thickness=2)
    

    roi = regionOF(edges, vertices)
    

    lines = houghLines(roi, 1, np.pi / 180, 15, 10, 20)  
    
    
    direction = directionDET(lines, width)
    print(f"Determined direction: {direction}")
    

    drawLines(birdseye, lines, direction)
    
    return birdseye

if __name__ == "__main__":
 
    cap = cv2.VideoCapture('/Users/charliefraser/Downloads/IMG_0897.MOV')

  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('tdfijd.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        processedFrame = processed(frame)
        
    
        out.write(processedFrame)
        
     
        cv2.imshow('Lane Detection', processedFrame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    cap.release()
    out.release()
    cv2.destroyAllWindows()
