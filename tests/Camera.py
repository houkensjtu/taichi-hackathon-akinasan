import cv2 as cv
from Processor import Processor

class Camera(object):
    def __init__(self, resolution, framerate):
        self.resolution = resolution
        self.framerate = framerate
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            print('Cannot open camera')
            exit()

    def start_recording(self, Processor):
        self.processor = Processor()
        frame_id = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print('Cannot receive frame. Exiting ...')
                break
            filename = f'{frame_id:06d}.jpg'
            cv.imwrite(filename, cv.resize(frame, self.resolution))
            print(f'>>> Captured and saved frame {filename}')
            frame_id += 1
            if cv.waitKey(1) == ord('q'):
                break

    def wait_recording(self, time):
        pass

    def stop_recording(self):
        # self.processor.flush()
        self.cap.release()
        cv.destroyAllWindows()
    
if __name__ == '__main__':
    dim = (160, 120)
    fps = 15
    camera = Camera(dim, fps)
    print('Start recording ...')
    camera.start_recording(Processor)
    sleep(2)

    camera.stop_recording()
    print('Finish recording ...')
