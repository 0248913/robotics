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

        # Apply Gaussian blur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply edge detection with fine-tuned parameters
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Detect lines using HoughLinesP with fine-tuned parameters
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=20)

        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Determine if the line is on the left or right based on x-coordinates
                if x1 < output_width / 2 and x2 < output_width / 2:
                    left_lines.append(line[0])
                elif x1 > output_width / 2 and x2 > output_width / 2:
                    right_lines.append(line[0])

        if left_lines and right_lines:
            # Calculate the average line for left and right lines
            left_avg_line = np.mean(left_lines, axis=0).astype(int)
            right_avg_line = np.mean(right_lines, axis=0).astype(int)

            # Draw the left and right average lines
            cv2.line(birdseye_view, (left_avg_line[0], left_avg_line[1]), (left_avg_line[2], left_avg_line[3]), (0, 0, 255), 2)
            cv2.line(birdseye_view, (right_avg_line[0], right_avg_line[1]), (right_avg_line[2], right_avg_line[3]), (0, 0, 255), 2)

            # Calculate the midpoints between corresponding endpoints of the left and right lines
            mid_start_x = (left_avg_line[0] + right_avg_line[0]) // 2
            mid_start_y = (left_avg_line[1] + right_avg_line[1]) // 2
            mid_end_x = (left_avg_line[2] + right_avg_line[2]) // 2
            mid_end_y = (left_avg_line[3] + right_avg_line[3]) // 2

            # Draw the middle line
            cv2.line(birdseye_view, (mid_start_x, mid_start_y), (mid_end_x, mid_end_y), (255, 0, 0), 2)

        # Display the original frame and the bird's-eye view with detected lines
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

