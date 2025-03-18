import cv2
import numpy as np

class PolygonAnnotator:
    def __init__(self, image1_path, image2_path):
        self.image1 = cv2.imread(image1_path)
        self.image2 = cv2.imread(image2_path)
        self.points1 = []
        self.points2 = []
        self.current_image = 1
        
        # Create windows
        cv2.namedWindow('Image 1')
        cv2.namedWindow('Image 2')
        
        # Set mouse callback
        cv2.setMouseCallback('Image 1', self.mouse_callback1)
        cv2.setMouseCallback('Image 2', self.mouse_callback2)
        
    def mouse_callback1(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points1.append([x, y])
            self.draw_points(self.image1, self.points1, False)
            cv2.imshow('Image 1', self.image1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points1) >= 3:
                self.draw_points(self.image1, self.points1, True)
                cv2.imshow('Image 1', self.image1)

    def mouse_callback2(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points2.append([x, y])
            self.draw_points(self.image2, self.points2, False)
            cv2.imshow('Image 2', self.image2)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points2) >= 3:
                self.draw_points(self.image2, self.points2, True)
                cv2.imshow('Image 2', self.image2)

    def draw_points(self, image, points, close_polygon):
        # Draw points and lines
        for i in range(len(points)):
            cv2.circle(image, tuple(points[i]), 3, (0, 255, 0), -1)
            if i > 0:
                cv2.line(image, tuple(points[i-1]), tuple(points[i]), (0, 255, 0), 2)
        # Draw line from last point to first point only when right-clicked
        if close_polygon and len(points) >= 3:
            cv2.line(image, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)

    def run(self):
        while True:
            cv2.imshow('Image 1', self.image1)
            cv2.imshow('Image 2', self.image2)
            
            key = cv2.waitKey(1) & 0xFF
            # Press 'q' to quit and print coordinates
            if key == ord('q'):
                print("\nPolygon coordinates for Image 1:")
                print(self.points1)
                print("\nPolygon coordinates for Image 2:")
                print(self.points2)
                break
            # Press 'c' to clear all points
            elif key == ord('c'):
                self.points1 = []
                self.points2 = []
                self.image1 = cv2.imread(image1_path)
                self.image2 = cv2.imread(image2_path)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace these paths with your actual image paths
    image1_path = "./simulationmaps.jpg"
    image2_path = "./cam2.jpg"
    
    annotator = PolygonAnnotator(image1_path, image2_path)
    annotator.run()
