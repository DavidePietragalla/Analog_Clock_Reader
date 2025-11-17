import cv2
import numpy as np
import math

def compute_angle(center, tip):
    """
    Compute the angle of a clock hand in radians.
    math.atan2() returns a value between -pi and +pi.
    The input (y, x) are the coordinates of the tip from the origin.
    """
    return math.atan2(tip[1] - center[1], tip[0] - center[0])

def angle_to_clock_value(angle_rad, hand_type):
    """
    Convert the angle (expressed in radians) to a clock value.
    hand_type determines if it's hour, minute or second hand.
    """
    # Convert to degrees
    angle_deg = math.degrees(angle_rad)
    
    # 1. Let's say the hour we as humans see is 3 o'clock
    # 2. The angle degree captured by atan2 would be 0 degrees (due to OpenCV coordinate system)
    # 3. For a clock, we need 0 degrees to be at the positive y-axis
    # 4. To perform the origin shift we add 90 degrees: (angle_deg + 90) % 360
    angle_clock = (angle_deg + 90) % 360
    
    value = 0
    if hand_type in ['second', 'minute']:
        value = angle_clock / 6  # 6 comes from 360 degrees / 60 units
    elif hand_type == 'hour':
        value = angle_clock / 30  # 30 comes from 360 degrees / 12 units
        
    return value

def detect_time(image_path):
    """
    Given the path to an image of a clock, detect the time shown on it.
    1. Preprocess the image and define the center.
    2. Detect edges and lines.
    3. Filter and identify clock hands.
    4. Assign hands and compute the time.
    5. Final calculation.
    6. Display the result.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Load the output quadrant
    quadrant_output = cv2.imread("clock.jpg")
    if quadrant_output is None:
        print(f"Error: Unable to load image of the clock quadrant")
        return

    # Resize for consistent processing
    img = cv2.resize(img, (500, 500))
    img_output = img.copy()
    height, width, _ = img.shape

    # --- 1. Preprocess the image and define the center ---
    center = (width // 2, height // 2)
    cx, cy = center
    cv2.circle(img_output, center, 5, (0, 0, 255), -1) # Draw the center

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0) # (7, 7) is the kernel size
    
    # --- 2. Detect edges and lines ---
    # In Canny(blurred, a, b), a and b are the min and max thresholds
    # Below the min threshold, edges are discarded
    # Above the max threshold, edges are accepted
    # Between min and max, edges are accepted if connected to strong edges
    edges = cv2.Canny(blurred, 50, 150)

    # HoughLinesP returns line segments in the form of endpoints (x1, y1, x2, y2)
    # Parameters:
    # Distance between points in pixels
    # Angle resolution in radians (np.pi/180 = 1 degree)
    # threshold: The smallest number of intersections required to consider a line segment valid
    # minLineLength: Line segments shorter than this value are rejected
    # maxLineGap: The maximum distance in pixels between line segments that can be merged to form a longer line
    min_radius = int(height * 0.1) # Idea: hour hand is at least 10% of height
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 
                            threshold=40, 
                            minLineLength=min_radius, 
                            maxLineGap=10)
    
    # Check if any lines were detected and show edges for debugging if not
    if lines is None:
        print("No lines found. Try adjusting Canny or Hough parameters.")
        cv2.imshow("Edges", edges)
        cv2.waitKey(0)
        return

    # --- 3. Filter and identify clock hands ---
    candidate_hands = []  # List of (length, tip, angle)
    
    # Max distance of the "base" of the hand from the center
    center_threshold = int(height * 0.1) # 10% of the height

    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate the distance of both endpoints from the center
        dist1 = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
        dist2 = np.sqrt((x2 - cx)**2 + (y2 - cy)**2)
        
        # Find the base point (closer to center) and the tip point (farther)
        if dist1 < dist2:
            base, tip, length = (x1, y1), (x2, y2), dist2
        else:
            base, tip, length = (x2, y2), (x1, y1), dist1
            
        # The base must be reasonably close to the center
        if np.sqrt((base[0] - cx)**2 + (base[1] - cy)**2) < center_threshold:
            angle = compute_angle(center, tip)
            candidate_hands.append((length, tip, angle))

    if not candidate_hands:
        print("No candidate hands found due to proximity to center.")
        return

    # Sort candidates by length (longest first)
    candidate_hands.sort(key=lambda x: x[0], reverse=True)

    # To avoid detecting the same hand multiple times, use angular difference filtering
    # (NOT SURE)
    final_hands = [] 
    min_angular_diff = 0.2

    for hand in candidate_hands:
        unique = True
        for final in final_hands:
            # Compute angular difference
            diff = abs(hand[2] - final[2])
            angular_diff = min(diff, 2 * np.pi - diff) 
            
            if angular_diff < min_angular_diff:
                unique = False
                break
        if unique:
            final_hands.append(hand)

    # final_hands now contains our hands
    # Sort by length (shortest to longest) to identify them
    final_hands.sort(key=lambda x: x[0])
    
    # --- 4. Assign hands and compute the time ---
    hands = {}
    if len(final_hands) == 2:
        # Assume Hours and Minutes
        hands['hour'] = final_hands[0]
        hands['minute'] = final_hands[1]
    elif len(final_hands) >= 3:
        # Assume Hours, Minutes, Seconds
        # If there are more than 3, take the 3 longest
        hands['hour'] = final_hands[-3]
        hands['minute'] = final_hands[-2]
        hands['second'] = final_hands[-1]
    elif len(final_hands) == 1:
        # Only one hand detected, assume hours and minutes hand are overlapping
        hands['hour'] = final_hands[0]
        hands['minute'] = final_hands[0]
    else:
        print(f"Only {len(final_hands)} unique hands found. Unable to determine time.")
        print(final_hands[0])
        return

    # --- 5. Final calculation ---
    hour_f, min_f, sec_f = 0.0, 0.0, 0.0

    if 'second' in hands:
        length, tip, angle = hands['second']
        sec_f = angle_to_clock_value(angle, 'second')
        cv2.line(img_output, center, tip, (255, 0, 0), 2) # Blue = Seconds
        cv2.line(quadrant_output, center, tip, (255, 0, 0), 2)
    if 'minute' in hands:
        length, tip, angle = hands['minute']
        min_f = angle_to_clock_value(angle, 'minute')
        cv2.line(img_output, center, tip, (0, 255, 0), 2) # Green = Minutes
        cv2.line(quadrant_output, center, tip, (0, 255, 0), 2)
    if 'hour' in hands:
        length, tip, angle = hands['hour']
        hour_f = angle_to_clock_value(angle, 'hour')
        cv2.line(img_output, center, tip, (0, 0, 255), 2) # Red = Hours
        cv2.line(quadrant_output, center, tip, (0, 0, 255), 2)

    # We need to handle the minutes for a more accurate estimate
    # Example:
    # If it's 10:50, hour_f will be ~10.83 and min_f will be ~50.
    # If it's 10:10, hour_f will be ~10.16 and min_f will be ~10.
    # If the minute hand is in the first half (0–30),
    # the read hour 10.8 is wrong, it should actually be 9.8.
    # Idea:
    # The hour hand angle depends on the minute hand.
    # True_Hour_Angle = (H % 12 + M / 60) * 30
    # Read_Hour_Angle = hour_f * 30

    # Round the final values
    hour_f = int(hour_f)
    min_f = int(min_f)
    sec_f = int(round(sec_f))
    
    # Overflow handling for seconds and minutes
    if sec_f == 60:
        sec_f = 0
        min_f += 1
        if min_f == 60:
            min_f = 0
            hour_f = (hour_f+1 % 12)
    if hour_f == 0:
        hour_f = 12

    # --- 6. Display the result ---
    res_str = f"{hour_f:02d}:{min_f:02d}:{sec_f:02d}"
    print(f"Time Detected: {res_str}")

    cv2.putText(img_output, res_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5) # Black outline
    cv2.putText(img_output, res_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # White text

    cv2.imshow(f"Detected Clock - {image_path}", img_output)
    cv2.imshow("Clock Quadrant Output", quadrant_output)
    cv2.imshow("Edges", edges)
    # cv2.waitKey(0)

    # Close when a key is pressed or window is closed
    while True:
        key = cv2.waitKey(100)
        if key != -1:
            break
        if cv2.getWindowProperty(f"Detected Clock - {image_path}", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    return res_str

# Call the function
path_to_image = "ora2.png"
detect_time(path_to_image)