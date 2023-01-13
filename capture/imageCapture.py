# import the opencv library
import cv2
import time

# define a video capture object
vid = cv2.VideoCapture(0)
# start time
start = time.time()
while(True):
    # read the video capture
    ret, frame = vid.read()
    # show the camera
    cv2.imshow('frame', frame)
    # check if the vid is opened to take the last frame
    if vid.isOpened() and ((time.time() - start) > 5):
        # save the frame
        cv2.imwrite("frame.jpg", frame)
        break
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
