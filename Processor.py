import cv2 as cv

class Processor(object):
    def __init__(self):
        self.frame_num = 0

    def write(self,buf):
        # 例如将JPEG帧保存成文件
        filename = f'{self.frame_num:06d}.jpg'
        cv.imwrite(filename, buf)
        print(f'>>> Captured and saved frame {filename}')
        self.frame_num += 1

    def flush(self):
        pass
