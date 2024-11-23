from datetime import datetime
import csv
import os

class AttendanceSystem:
    def __init__(self, attendance_file="attendance.csv"):
        self.attendance_file = attendance_file
        self.initialize_attendance_file()

    def initialize_attendance_file(self):
        # Create attendance file if it doesn't exist
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'Time'])

    
    def person_exists_today(self, name):
        """Check if person already exists in today's records"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if row and row[0] == name and row[1] == current_date:
                        return True
        return False

    def mark_attendance(self, name):
        if name != "Unknown":  
           # if not self.person_exists_today(name):
                current_date = datetime.now().strftime("%Y-%m-%d")
                current_time = datetime.now().strftime("%H:%M:%S")
                
                with open(self.attendance_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, current_date, current_time])