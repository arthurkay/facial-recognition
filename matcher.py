import cv2
import face_recognition
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--test", required=True, type=str, help="Provide path to comparison image")
ap.add_argument("-s", "--sample", required=True, type=str, help="Provide an image to sample against")
ap.add_argument("-n", "--name", type=str, help="You can provide an optional name of person in image")
args = vars( ap.parse_args() )

imgSample = face_recognition.load_image_file(args["sample"])
imgTest = face_recognition.load_image_file(args["test"])

rgbSample = cv2.cvtColor(imgSample, cv2.COLOR_BGR2RGB)
rgbTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Get test image face locations
face_loc = face_recognition.face_locations(rgbTest)[0]
face_loc_match = face_recognition.face_locations(rgbSample)[0]

# Create image encodings on the face
trainEncoding = face_recognition.face_encodings(imgSample)[0]
unknown = face_recognition.face_encodings(imgTest)[0]

cv2.putText(rgbSample, "Matching...", (face_loc_match[3], face_loc_match[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
cv2.rectangle(rgbSample, (face_loc_match[3], face_loc_match[0]),(face_loc_match[1], face_loc_match[2]), (0, 255, 255), 1)
cv2.imshow("Train Image", rgbSample)

result = face_recognition.compare_faces([trainEncoding], unknown)
relativeness = face_recognition.face_distance([trainEncoding], unknown)
print("Are the images the same? {} \nWhats the probability of relativenes? {}".format(result[0], relativeness))

if result[0]:
    if args["name"]:
        cv2.putText(rgbTest, args["name"], (face_loc[3], face_loc[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.rectangle(rgbTest, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (0, 255, 0), 1)
    else:
        cv2.putText(rgbTest, "Matched", (face_loc[3], face_loc[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.rectangle(rgbTest, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (0, 0, 255), 1)
else:
    cv2.putText(rgbTest, "Not a Match", (face_loc[3], face_loc[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.rectangle(rgbTest, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (0, 0, 255), 1)

cv2.imshow("Test Image", rgbTest)

if cv2.waitKey(0) == ord("q"):
    cv2.destroyAllWindows()