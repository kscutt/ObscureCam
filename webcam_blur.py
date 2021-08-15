# Coded by Kathryn Scutt for TecHacks2.0
# This program uses python and OpenCV to detect a face
# and blur the face and the background in two
# separate windows.
# I am using a usb webcam and not the one included on my laptop
import cv2

# Create a cascade to recognize the face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Capture the webcam, I use 1 because that is my usb webcam
webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Set dimensions for the webcam
webcam.set(3, 960)
webcam.set(4, 800)


# This function is used to blur the background
def blur_img(img, factor):
    kW = int(img.shape[1] / factor)
    kH = int(img.shape[0] / factor)

    # ensure the shape of the kernel is odd
    if kW % 2 == 0: kW = kW - 1
    if kH % 2 == 0: kH = kH - 1

    blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
    return blurred_img


# Create a loop that displays the webcam feed until
# q is pressed
while True:
    success, img = webcam.read()
    # define image to blur and by what factor
    # higher factor = more blur
    blurred_img = blur_img(img, factor=10)

    # convert to greyscale because OpenCV expects it
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(
        grey,
        scaleFactor=1.3,
        minSize=(30, 30),
        minNeighbors=5,

    )

    for (x, y, w, h) in face:
        # The padding is just to make the square bigger
        padding = 20

        # define the area that is the detected face
        detected_face = img[int(y):int(y + h), int(x):int(x + w)]

        blurred_img[y:y + h, x:x + w] = detected_face
        # Draw a rectangle around the faces
        cv2.rectangle(blurred_img, (x - padding, y - padding), (x + w + padding, y + h + padding), (0, 255, 0), 2)
        cv2.rectangle(img, (x - padding, y - padding), (x + w + padding, y + h + padding), (255, 255, 255), 2)

        # blur the face
        img[y:y + h, x:x + w] = cv2.medianBlur(img[y:y + h, x:x + w], 35)

    # Show the blurred face cam
    cv2.imshow("Blur Face", img)

    # Show the blurred background cam
    cv2.imshow("Blur Background", blurred_img)

    # adds a delay and looks for key press q to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close everything
webcam.release()
cv2.destroyAllWindows()
