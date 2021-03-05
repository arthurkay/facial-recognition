import cv2
import face_recognition
import argparse

'''
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sample", required=True,
help="Provide a sample image to test against")
args = vars( ap.parse_args() )
'''

sampleImg = "images/train/Katema.jpg"

# Get image darray
imgArray = face_recognition.load_image_file(sampleImg)

# Create image encodings
ArthurEncoding = face_recognition.face_encodings(imgArray)

vidStream = cv2.VideoCapture(0)

if not vidStream.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = vidStream.read()

    # Get frame in rgb
    rgbFrame = frame[:, :, ::-1]
    
    # Face locations
    face_loc = face_recognition.face_locations(rgbFrame)

    # Img encodings
    faceEncodings = face_recognition.face_encodings(rgbFrame, face_loc)

    for face_encoding in faceEncodings:
        # Draw rectabgle on face
        cv2.rectangle(rgbFrame, (face_loc[3], face_loc[0]), (face_loc[0], face_loc[2]), (0, 255, 0), 1)
        result = face_recognition.compare_faces(ArthurEncoding, face_encoding)
        print("The person in the image is Arthur? {}".format(result[0]))

    cv2.imshow("Video Stream", rgbFrame)
    cv2.imshow("Video Stream 2", frame)

    c = cv2.waitKey(1)
    # Pres esc key to exit
    if c == 27:
        break

vidStream.release()
cv2.destroyAllWindows()
