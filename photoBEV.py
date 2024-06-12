import cv2
import numpy as np

def imageBEV(imagePath):
   
    frame = cv2.imread(imagePath)
    if frame is None:
        print("Error: Could not load image.")
        return

    output_width = 320
    output_height = 240

    
    pts1 = np.float32([
        [0, 1080],     # Bottom-left
        [1920, 1080],  # Bottom-right
        [1632, 648],   # Top-right
        [288, 648]     # Top-left
    ])

   
    pts2 = np.float32([
        [0, output_height],          
        [output_width, output_height], 
        [output_width, 0],            
        [0, 0]                        
    ])

  
    M = cv2.getPerspectiveTransform(pts1, pts2)
    birdseye_view = cv2.warpPerspective(frame, M, (output_width, output_height))

   
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Birds Eye View', birdseye_view)

   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    imagePath = 'captured_image.jpg'  
    imageBEV(imagePath)

