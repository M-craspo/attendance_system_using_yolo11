import cv2
import os
import numpy as np
import face_recognition
from ultralytics import YOLO
from typing import Dict, List, Tuple
import time
class SimpleFacerec:
    def __init__(self, 
                 frame_resizing=0.7,
                 threshold=0.8,
                 person_model_path="yolo11n.pt",
                 model_confidence=0.5):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = frame_resizing
        self.threshold = threshold
        self.person_model = YOLO(person_model_path)
        self.model_confidence = model_confidence

    def load_encoding_images(self, images_path):
        """Load known face encodings and names from a directory."""
        if isinstance(images_path, str):
            # Handle directory path
            image_files = [os.path.join(images_path, f) for f in os.listdir(images_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        elif isinstance(images_path, dict):
            # Handle dictionary of name:path pairs
            image_files = list(images_path.values())
            
        print(f"{len(image_files)} encoding images found.")

        for img_path in image_files:
            print(f"Processing image: {img_path}")

            try:
                # Load image using face_recognition for better compatibility
                img = face_recognition.load_image_file(img_path)

                # Get face encodings
                encodings = face_recognition.face_encodings(img, model="large")

                if encodings:
                    # Extract name from file name or dictionary key
                    if isinstance(images_path, dict):
                        name = [k for k, v in images_path.items() if v == img_path][0]
                    else:
                        name = os.path.splitext(os.path.basename(img_path))[0]
                        
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(name)
                    print(f"Successfully encoded {name}")
                else:
                    print(f"No face detected in {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

    def detect_known_faces(self, frame):
        """Detect known faces in a frame using YOLO for person detection."""
        # Resize the frame
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        
        # Convert to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = []
        face_names = []
        
        try:
            # Detect persons using YOLO
            results = self.person_model.predict(small_frame, conf=self.model_confidence, classes=[0])
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get person box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Extract person region
                    person_region = rgb_small_frame[y1:y2, x1:x2]
                    
                    # Detect faces in the person region
                    person_face_locations = face_recognition.face_locations(person_region)
                    if person_face_locations:
                        face_encodings = face_recognition.face_encodings(person_region, 
                                                                       person_face_locations, 
                                                                       model="large")
                        
                        for (top, right, bottom, left), face_encoding in zip(person_face_locations, face_encodings):
                            # Compare with known faces
                            matches = face_recognition.compare_faces(self.known_face_encodings, 
                                                                  face_encoding)
                            name = "Unknown"
                            
                            if True in matches:
                                # Calculate face distances
                                face_distances = face_recognition.face_distance(self.known_face_encodings, 
                                                                             face_encoding)
                                best_match_index = np.argmin(face_distances)
                                min_distance = face_distances[best_match_index]
                                
                                # Only consider it a match if the distance is below the threshold
                                if matches[best_match_index] and min_distance < self.threshold:
                                    name = self.known_face_names[best_match_index]
                            
                            # Adjust face location to account for person region
                            adjusted_location = (
                                top + y1, right + x1,
                                bottom + y1, left + x1
                            )
                            face_locations.append(adjusted_location)
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