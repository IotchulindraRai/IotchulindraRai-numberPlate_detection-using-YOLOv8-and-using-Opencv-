import cv2
import pytesseract
import datetime
import csv

# Load the license plate detector
plate_cascade = cv2.CascadeClassifier('model/haarcascade_russian_plate_number.xml')

# Set up the camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Set the minimum license plate area and the delay between scans (in seconds)
min_area = 500
scan_delay = 1

# Set up the CSV file
today = datetime.date.today()
filename = f'plates_{today}.csv'
header = ['Plate Number', 'Timestamp']
with open(filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

# Start scanning
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale and detect license plates
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, 1.1, 4)

    # Check each license plate
    for (x, y, w, h) in plates:
        area = w * h

        # Ignore small license plates
        if area < min_area:
            continue

        # Extract the license plate image and apply OCR
        plate_img = gray[y:y + h, x:x + w]
        plate_text = pytesseract.image_to_string(plate_img, config='--psm 7')

        # Ignore license plates that don't contain both letters and numbers
        if not any(c.isalpha() for c in plate_text) or not any(c.isdigit() for c in plate_text):
            continue

        # Ignore license plates with less than 6 alphanumeric characters
        alphanumeric = ''.join(c for c in plate_text if c.isalnum())
        if len(alphanumeric) < 6:
            continue

        # Write the license plate number and timestamp to the CSV file
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([plate_text, timestamp])

        # Draw a rectangle around the license plate and display the image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, plate_text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

    # Display the image and wait for the specified delay
    cv2.imshow('License Plate Detector', frame)

    if cv2.waitKey(scan_delay * 1000) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
