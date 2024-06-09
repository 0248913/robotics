import cv2
import numpy as np

def apply_birdseye_view_to_live_feed():
    # Initialize USB webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define the dimensions of the output bird's-eye view image
    output_width, output_height = 320, 240

    # Define the source points (trapezoid) in the original image
    pts1 = np.float32([
        [0, 1080],      # Bottom-left
        [1920, 1080],   # Bottom-right
        [1632, 648],    # Top-right
        [288, 648]      # Top-left
    ])

    # Define the destination points (rectangle) in the output bird's-eye view image
    pts2 = np.float32([
        [0, output_height],            # Bottom-left
        [output_width, output_height], # Bottom-right
        [output_width, 0],             # Top-right
        [0, 0]                         # Top-left
    ])

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        # Apply the perspective transformation to get the bird's-eye view
        birdseye_view = cv2.warpPerspective(frame, M, (output_width, output_height))

        # Convert to grayscale
        gray = cv2.cvtColor(birdseye_view, cv2.COLOR_BGR2GRAY)

        # Apply edge detection with fine-tuned parameters
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using HoughLinesP with fine-tuned parameters
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=20)

        if lines is not None:
            # List to store detected lines for further processing
            detected_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(birdseye_view, (x1, y1), (x2, y2), (0, 255, 0), 2)
                detected_lines.append(((x1, y1), (x2, y2)))

            # Function to calculate the distance between two lines
            def calculate_distance(line1, line2):
                ((x1, y1), (x2, y2)) = line1
                ((x3, y3), (x4, y4)) = line2
                # Calculate the midpoints of the two lines
                midpoint1 = ((x1 + x2) // 2, (y1 + y2) // 2)
                midpoint2 = ((x3 + x4) // 2, (y3 + y4) // 2)
                # Calculate the Euclidean distance between the midpoints
                distance = np.sqrt((midpoint1[0] - midpoint2[0]) ** 2 + (midpoint1[1] - midpoint2[1]) ** 2)
                return distance, midpoint1, midpoint2

            # Iterate through detected lines to find and display the distance
            for i in range(len(detected_lines) - 1):
                for j in range(i + 1, len(detected_lines)):
                    distance, midpoint1, midpoint2 = calculate_distance(detected_lines[i], detected_lines[j])
                    # Draw a line between the midpoints
                    cv2.line(birdseye_view, midpoint1, midpoint2, (255, 0, 0), 2)
                    # Display the distance
                    cv2.putText(birdseye_view, f'{distance:.2f}', ((midpoint1[0] + midpoint2[0]) // 2, (midpoint1[1] + midpoint2[1]) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the original frame and the bird's-eye view with detected lines and distances
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Birds Eye View with Line Detection', birdseye_view)

        # Break the loop on 'ESC' key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    apply_birdseye_view_to_live_feed()
