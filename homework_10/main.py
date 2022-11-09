import cv2

path = 'resources/dashcam.mp4'
video = cv2.VideoCapture(path)

# define the bounding box of a detected car
x1, y1 = 595, 365
x2, y2 = 645, 410
width = x2 - x1
height = y2 - y1

# chose tracker
tracker_types = ['MIL', 'CSRT', 'MOSSE']
tracker_type = tracker_types[0]

# set up tracker
if tracker_type == 'MIL':
    # MIL algorithm trains a classifier in an online manner to separate the object from the background
    tracker = cv2.TrackerMIL_create()
elif tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()
elif tracker_type == "MOSSE":
    # Minimum Output Sum of Squared Error tracker
    tracker = cv2.legacy.TrackerMOSSE_create()
else:
    tracker = cv2.TrackerCSRT_create()

first_frame = video.read()[1]

# initialize tracker
bbox = (x1, y1, width, height)
ok = tracker.init(first_frame, bbox)

# tracking sequence
while video.isOpened():
    # read frame
    present, frame = video.read()
    if present:
        # find new position of the object
        ok, bbox = tracker.update(frame)
        print(ok, bbox)
        # draw bounding box
        x1, y1 = int(bbox[0]), int(bbox[1])
        width, height = int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
        cv2.waitKey(5)
    else:
        break

# close video file
video.release()
# close all frames
cv2.destroyAllWindows()

# Conclusion:
# MIL:
#   Advantages:
#   - keeps tracking an object if perspective changes
#   - keeps tracking an object if it changes its size
#   - resistant to luminosity changes
#   Disadvantages:
#   - very slow (the slowest among three)
#   - cannot adjust the bounding box if object's scale changes
#   - keeps tracking random objects when the main object goes off the screen
# CSRT:
#   Advantages:
#   - keeps tracking an object if perspective changes
#   - keeps tracking an object if it changes its size
#   - adjusts the bounding box if object's scale changes
#   - resistant to luminosity changes
#   - stops tracking when the main object goes off the screen
#   Disadvantages:
#   - quite slow (but faster than MIL)
#   - fails to adjust the bounding box if object's scale changes too rapidly
# MOSSE:
#   Advantages:
#   - very fast
#   - keeps tracking an object if perspective changes
#   - keeps tracking an object if it changes its size
#   - resistant to luminosity changes
#   - stops tracking when the main object goes off the screen
#   Disadvantages:
#   - cannot adjust the bounding box if object's scale changes
