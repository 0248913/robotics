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

        if lines is not None:
            # Filter and merge lines
            merged_lines = merge_lines(lines)

            for line in merged_lines:
                x1, y1, x2, y2 = line
                cv2.line(birdseye_view, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

def merge_lines(lines, min_distance=10, min_angle=5):
    """
    Merge lines that are close to each other and have a similar angle.
    """
    def compute_distance_and_angle(line1, line2):
        """
        Compute the distance between the midpoints of two lines and the angle difference.
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Calculate midpoints
        mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)

        # Calculate distance between midpoints
        distance = np.sqrt((mid1[0] - mid2[0]) ** 2 + (mid1[1] - mid2[1]) ** 2)

        # Calculate angles of each line
        angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angle2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi

        # Calculate angle difference
        angle_diff = abs(angle1 - angle2)

        return distance, angle_diff

    def should_merge(line1, line2):
        """
        Determine if two lines should be merged based on their distance and angle.
        """
        distance, angle_diff = compute_distance_and_angle(line1, line2)
        return distance < min_distance and angle_diff < min_angle

    merged_lines = []

    for line in lines:
        line = line[0]  # Convert from 2D array to 1D array
        if len(merged_lines) == 0:
            merged_lines.append(line)
        else:
            merged = False
            for i, existing_line in enumerate(merged_lines):
                if should_merge(existing_line, line):
                    # Merge lines by averaging the endpoints
                    x1, y1, x2, y2 = line
                    ex1, ey1, ex2, ey2 = existing_line
                    new_line = [
                        int((x1 + ex1) / 2),
                        int((y1 + ey1) / 2),
                        int((x2 + ex2) / 2),
                        int((y2 + ey2) / 2)
                    ]
                    merged_lines[i] = new_line
                    merged = True
                    break
            if not merged:
                merged_lines.append(line)

    return merged_lines

if __name__ == '__main__':
    apply_birdseye_view_to_live_feed()
