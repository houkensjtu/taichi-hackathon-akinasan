class Camera:
    def __init__(self, dim, fps):
        self.dim = dim
        self.fps = fps
        self.cap = cv.VideoCapture(0)

    def start_recording(self, Operation):
        pass
