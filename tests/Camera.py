import cv2 as cv
from Processor import Processor

class Camera(object):
    def __init__(self, resolution, framerate, vflip, hflip):
        self.resolution = resolution
        self.framerate = framerate  # Not used; see https://stackoverflow.com/questions/52068277/change-frame-rate-in-opencv-3-4-2
        self.cap = cv.VideoCapture(0)
        self.vflip = vflip
        self.hflip = hflip
        if not self.cap.isOpened():
            print('Cannot open camera')
            exit()

    def start_recording(self, Processor):
        self.processor = Processor()
        ret, frame = self.cap.read()  # Read one frame to test the camera
        if not ret:
            print('Cannot receive frame. Exiting ...')
        

    def wait_recording(self, num_of_frames):  # Record a certain number of frames
        while num_of_frames > 0:
            ret, frame = self.cap.read()
            
            # Image processing included here
            # Because resolution and flip parameters are inside the Camera class
            resized_frame = cv.resize(frame, self.resolution)
            if self.vflip:
                frame = cv.flip(resized_frame, 0)
            if self.hflip:
                frame = cv.flip(resized_frame, 1)
                
            self.processor.write(frame)
            if cv.waitKey(1) == ord('q'):
                break
            num_of_frames -= 1
    

    def stop_recording(self):
        self.processor.flush()
        self.cap.release()
        cv.destroyAllWindows()
        
    
if __name__ == '__main__':
    dim = (160, 120)
    fps = 15  # Not used; dummy variable
    camera = Camera(resolution=dim, framerate=fps, vflip=True, hflip=True)
    print('Start recording ...')
    camera.start_recording(Processor)
    camera.wait_recording(10)  # Record 10 framesae
    camera.stop_recording()
    print('Finish recording ...')
