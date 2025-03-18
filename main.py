import cv2
import json
import numpy as np
from data import camconfig

# Load videos
video_paths = [
    "./media/cam1.mp4",
    "./media/cam2.mp4",
    "./media/cam3.mp4",
    "./media/cam4.mp4",
]


# Calculate homography matrix for each camera
for camera_key in camconfig:
    src_pts = np.array(camconfig[camera_key]["flat_coordinates"], dtype=np.float32)
    dst_pts = np.array(camconfig[camera_key]["perspective_coordinates"], dtype=np.float32)
    H, _ = cv2.findHomography(dst_pts, src_pts)
    camconfig[camera_key]["homography_matrix"] = H.tolist()
    print(f"\nHomography matrix for {camera_key}:")
    print(H)

caps = [cv2.VideoCapture(path) for path in video_paths]

# Function to draw ROI polygon
def draw_roi(frame, coordinates, color):
    # Convert color string to BGR tuple
    color_str = color.replace("rgba(", "").replace(")", "").split(",")
    b, g, r = map(int, color_str[:3])
    alpha = float(color_str[3])

    # Create points array for polygon
    pts = np.array(coordinates, np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Draw filled polygon with transparency
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (b, g, r))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw polygon outline
    cv2.polylines(frame, [pts], True, (b, g, r), 2)

def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        if ((polygon[i][1] > y) != (polygon[j][1] > y) and
            (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
             (polygon[j][1] - polygon[i][1]) + polygon[i][0])):
            inside = not inside
        j = i
    
    return inside

# Load bounding box data from human_detection.json
with open('human_detection.json', 'r') as f:
    detection_data = json.load(f)

# Create a list of bounding boxes for each camera
bbox_data = []
for cam_idx in range(1, 5):  # For cameras 1-4
    camera_key = f'cam{cam_idx}'
    camera_boxes = []
    
    # Convert the polygon coordinates to bounding boxes
    for frame_idx, polygon in enumerate(detection_data[camera_key]):
        if polygon:  # Check if polygon exists for this frame
            x_coords = polygon[0::2]  # Get all x coordinates
            y_coords = polygon[1::2]  # Get all y coordinates
            
            if x_coords and y_coords:  # If coordinates exist
                bbox = {
                    "frame": frame_idx,
                    "bbox": {
                        "x_min": min(x_coords),
                        "y_min": min(y_coords),
                        "x_max": max(x_coords),
                        "y_max": max(y_coords)
                    }
                }
                camera_boxes.append(bbox)
    
    bbox_data.append(camera_boxes)

frame_idx = 0

# Load map image
map_image = cv2.imread("simulationmaps.jpg")

while True:
    frames = []
    map_display = map_image.copy()
    
    # Draw ROIs on map first
    for cap_idx, cap in enumerate(caps):
        camera_key = f'camera{cap_idx + 1}'
        if camera_key in camconfig:
            # Draw ROI polygon on map
            flat_coords = np.array(camconfig[camera_key]["flat_coordinates"], dtype=np.int32)
            color_str = camconfig[camera_key]["color"].replace('rgba(', '').replace(')', '').split(',')
            b, g, r = map(int, color_str[:3])
            alpha = float(color_str[3])
            
            # Create transparent overlay for ROI
            overlay = map_display.copy()
            cv2.fillPoly(overlay, [flat_coords], (b, g, r))
            cv2.addWeighted(overlay, alpha, map_display, 1 - alpha, 0, map_display)
            
            # Draw ROI outline
            cv2.polylines(map_display, [flat_coords], True, (b, g, r), 2)

    for cap_idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            break

        camera_key = f'camera{cap_idx + 1}'
        
        # Draw ROI for current camera view
        if camera_key in camconfig:
            draw_roi(
                frame,
                camconfig[camera_key]["perspective_coordinates"],
                camconfig[camera_key]["color"]
            )

        # Find bounding box for current frame if it exists
        bbox = None
        for data in bbox_data[cap_idx]:
            if data["frame"] == frame_idx:
                bbox = data["bbox"]
                break

        # Draw bounding box and project bottom point to map if found
        if bbox:
            x_min = bbox["x_min"]
            y_min = bbox["y_min"]
            x_max = bbox["x_max"]
            y_max = bbox["y_max"]

            # Draw bounding box on camera view
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Person",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Get bottom center point of bounding box
            bottom_center = np.array([[(x_min + x_max) // 2, y_max]], dtype=np.float32)
            
            # Transform point using homography matrix
            H = np.array(camconfig[camera_key]["homography_matrix"])
            transformed_point = cv2.perspectiveTransform(bottom_center.reshape(-1, 1, 2), H)
            map_point = transformed_point[0][0].astype(int)
            
            # Check if point is inside ROI before drawing
            flat_coords = camconfig[camera_key]["flat_coordinates"]
            if point_in_polygon(map_point, flat_coords):
                # Get camera color
                color_str = camconfig[camera_key]["color"].replace('rgba(', '').replace(')', '').split(',')
                b, g, r = map(int, color_str[:3])
                
                # Draw point on map with camera color
                cv2.circle(map_display, tuple(map_point), 5, (b, g, r), -1)
                cv2.putText(
                    map_display,
                    f"Cam{cap_idx + 1}",
                    (map_point[0] + 5, map_point[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (b, g, r),
                    2,
                )

        # Resize frame after drawing
        frame = cv2.resize(frame, (640, 360))
        frames.append(frame)

    if not frames:
        break

    # Combine frames into 2x2 grid
    top_row = np.hstack((frames[0], frames[1]))
    bottom_row = np.hstack((frames[2], frames[3]))
    combined = np.vstack((top_row, bottom_row))

    # Resize map_display to match the width of combined frame
    map_height = int(map_display.shape[0] * (combined.shape[1] / map_display.shape[1]))
    map_display = cv2.resize(map_display, (combined.shape[1], map_height))

    # Stack the map view below the camera views
    final_display = np.vstack((combined, map_display))

    # Display combined frame with map
    cv2.imshow("Multi-Camera View with Map", final_display)

    frame_idx += 1

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# Release resources
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
