import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_id =0
while True:
    dim = (160, 120)    
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    # cv.imshow('frame', gray)
    # cv.imshow('frame', frame)
    filename = f'{frame_id:06d}.jpg'
    cv.imwrite(filename, cv.resize(frame, dim))
    print(f'>>> Captured and saved frame {filename}')
    frame_id += 1
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
