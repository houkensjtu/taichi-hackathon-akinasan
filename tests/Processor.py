import cv2 as cv

class Processor(object):
    def __init__(self):
        self.frame_num = 0

    def write(self,buf):
        # 例如将JPEG帧保存成文件
        if buf.startswith(b'\xff\xd8'):
            if self.output:
                self.output.close()
            self.frame_num += 1
            self.output = io.open('{}_image{}.jpg'.format(key, time()), 'wb')
        self.output.write(buf)

    def flush(self):
        pass
