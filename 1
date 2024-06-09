import cv2
import numpy as np

def apply_birdseye_view_to_live_feed():
  
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

   
    output_width, output_height = 320, 240

   
    pts1 = np.float32([
        [0, 1080],      # Bottom-left
        [1920, 1080],   # Bottom-right
        [1632, 648],    # Top-right
        [288, 648]      # Top-left
    ])

   
    pts2 = np.float32([
        [0, output_height],          # Bottom-left
        [output_width, output_height], # Bottom-right
        [output_width, 0],            # Top-right
        [0, 0]                        # Top-left
    ])

    
    M = cv2.getPerspectiveTransform(pts1, pts2)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

      
        birdseye_view = cv2.warpPerspective(frame, M, (output_width, output_height))

      
        gray = cv2.cvtColor(birdseye_view, cv2.COLOR_BGR2GRAY)

      
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

       
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=20)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(birdseye_view, (x1, y1), (x2, y2), (0, 255, 0), 2)

      
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Birds Eye View with Line Detection', birdseye_view)

        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

   
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    apply_birdseye_view_to_live_feed()
