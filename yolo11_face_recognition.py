import cv2
import os
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
import time

class YOLOFaceRec:
    def __init__(self, 
                 frame_resizing=0.7,
                 face_model_path="yolo11n.pt",  # Path to YOLO face detection/recognition model
                 confidence_threshold=0.5):
        self.known_face_names = []
        self.frame_resizing = frame_resizing
        self.confidence_threshold = confidence_threshold
        self.face_model = YOLO(face_model_path)
        
    def load_encoding_images(self, images_path):
        """Load known face names from a directory or dictionary."""
        if isinstance(images_path, str):
            # Handle directory path
            image_files = [os.path.join(images_path, f) for f in os.listdir(images_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            # Extract names from filenames
            self.known_face_names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
        elif isinstance(images_path, dict):
            # Handle dictionary of name:path pairs
            self.known_face_names = list(images_path.keys())
            
        print(f"Loaded {len(self.known_face_names)} known face names.")
        
    def detect_known_faces(self, frame):
        """Detect and recognize faces in a frame using YOLO."""
        # Resize the frame
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        
        face_locations = []
        face_names = []
        
        try:
            # Detect faces using YOLO
            results = self.face_model.predict(small_frame, conf=self.confidence_threshold)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get face box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Get class prediction (if model is trained for face recognition)
                    if hasattr(box, 'cls'):
                        class_id = int(box.cls[0])
                        # If confidence is high enough and class_id is valid
                        if confidence >= self.confidence_threshold and class_id < len(self.known_face_names):
                            name = self.known_face_names[class_id]
                        else:
                            name = "Unknown"
                    else:
                        name = "Unknown"  # If model only does detection
                    
                    # Store face location
                    face_locations.append((y1, x2, y2, x1))
                    face_names.append(name)
                            
        except Exception as e:
            print(f"Error in face detection: {str(e)}")

        # Adjust face locations to the original frame size
        scale = 1 / self.frame_resizing
        face_locations = [(int(y1 * scale), int(x2 * scale), int(y2 * scale), int(x1 * scale)) 
                         for (y1, x2, y2, x1) in face_locations]

        return face_locations, face_names

    def process_video(self, source=0, display=True, max_duration=None):
        """Process video stream with optional duration limit."""
        cap = cv2.VideoCapture(source)
        start_time = time.time()

        try:
            while cap.isOpened():
                if max_duration and (time.time() - start_time) > max_duration:
                    break

                ret, frame = cap.read()
                if not ret:
                    break
                
                face_locations, face_names = self.detect_known_faces(frame)
                
                # Draw results
                for (y1, x2, y2, x1), name in zip(face_locations, face_names):
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw name
                    cv2.putText(frame, name, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                if display:
                    cv2.imshow('Face Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def process_image(self, image_path, display=True):
        """Process a single image."""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error loading image: {image_path}")
            return
        
        face_locations, face_names = self.detect_known_faces(frame)
        
        # Draw results
        for (y1, x2, y2, x1), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        if display:
            cv2.imshow('Face Recognition', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return frame
