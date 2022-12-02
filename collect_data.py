from ctlPWM import car_control
from Camera import Camera
import cv2 as cv
import os
os.environ['SDL_VIDEODRIVE'] = 'x11'
import pygame

import io
import time
import threading


global is_capture_running, key

class Processor(object):
    def __init__(self):
        self.frame_num = 0

    def write(self,buf):
        global key
        
        filename = '{}_image{:0>6d}.jpg'.format(key,self.frame_num)
        cv.imwrite(filename, buf)
        print(f'>>> Captured and saved frame {filename}')
        self.frame_num += 1

    def flush(self):
        pass

def cam_capture():
    global is_capture_running,key
    
    print("Start capture")        
    is_capture_running = True

    camera = Camera(resolution=(160, 120), framerate=30, vflip=False, hflip=False)
    outproc = Processor()

    camera.start_recording(outproc)
    camera.wait_recording(120)
    camera.stop_recording()

    print('Captured {} frames at {}fps'.format(outproc.frame_num, outproc.frame_num / 120.0))
    print("quit camera capture")
    is_capture_running = False

def start_drive(cc): 
    global is_capture_running, key
    key = 4
    pygame.init()
    pygame.display.set_mode((1,1))
    cc.car_stop()
    time.sleep(0.1)
    print("Start control!")
 
    while is_capture_running:
        # get input from human driver
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key_input = pygame.key.get_pressed()
                print(key_input[pygame.K_w], key_input[pygame.K_a], key_input[pygame.K_d])
                if key_input[pygame.K_w] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
                    print("Forward")
                    key = 2
                    cc.car_move_forward()
                elif key_input[pygame.K_a]:
                    print("Left")
                    cc.car_turn_left()
                    time.sleep(0.1)
                    key = 0
                elif key_input[pygame.K_d]:
                    print("Right")
                    cc.car_turn_right()
                    time.sleep(0.1)
                    key = 1
                elif key_input[pygame.K_s]:
                    print("Backward")
                    cc.car_move_backward()
                    key = 3
                elif key_input[pygame.K_k]:
                    cc.car_stop()
                    key = 4
            elif event.type == pygame.KEYUP:
                key_input = pygame.key.get_pressed()
                if key_input[pygame.K_w] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
                    print("Forward")
                    key = 2
                    cc.car_move_forward()
                elif key_input[pygame.K_s] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
                    print("Backward")
                    key = 3
                    cc.car_move_backward()
                else:
                    print("Stop")
                    cc.car_stop()
                    key = 4

if __name__ == '__main__':
    global is_capture_running, key

    print("capture thread")
    print ('-' * 50)
    capture_thread = threading.Thread(target=cam_capture,args=())
    capture_thread.setDaemon(True)
    capture_thread.start()
    
    speed = 0.5
    cc = car_control(speed)
    start_drive(cc)

    while is_capture_running:
        pass

    print("Done!")
    cc.clean_GPIO()
