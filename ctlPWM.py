import Jetson.GPIO as GPIO
from multiprocessing import Process
from multiprocessing import Queue
import time

class Signal(object):
    def __init__(self):
        self.run_flag = True
        self.frequency = 100 # 100HZ
        self.duty_cycle = 0.0

class PWM(object):
    def __init__(self,channel):
        self.pwm_flag = True
        self.channel = channel

    def pwm_start(self,queue):
        # initial parameters to prevent misoperation
        print('PWM start')
        print(self.pwm_flag,self.channel)
        f = 100.0  # default Frequency
        c = 0.0  # default duty cycle
        t = 1 / f
        t_h = t * c
        t_l = t * (1 - c)

        while self.pwm_flag:
            GPIO.output(self.channel, GPIO.HIGH)
            time.sleep(t_h)
            GPIO.output(self.channel, GPIO.LOW)
            time.sleep(t_l)

            if not queue.empty():
                n_sig = queue.get()
                self.pwm_flag = n_sig.run_flag
                f = n_sig.frequency
                c = n_sig.duty_cycle
                t = 1 / f
                t_h = t * c
                t_l = t * (1 - c)

class car_control(object):
    """docstring for ClassName"""
    def __init__(self,speed):
        self.backMotorInput1 = 13
        self.backMotorInput2 = 15
        self.backMotorEn = 12
        self.frontMotorInput1 = 7
        self.frontMotorInput2 = 11
        self.frontMotorEn = 16

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.backMotorInput1, GPIO.OUT)
        GPIO.setup(self.backMotorInput2, GPIO.OUT)
        GPIO.setup(self.backMotorEn, GPIO.OUT)
        GPIO.setup(self.frontMotorEn, GPIO.OUT)
        GPIO.setup(self.frontMotorInput1, GPIO.OUT)
        GPIO.setup(self.frontMotorInput2, GPIO.OUT)

        self.pwm = PWM(self.backMotorEn)
        self.sig = Signal()
        self.sig.duty_cycle = speed
        self.q = Queue()
        self.q.put(self.sig)
        self.p = Process(target=self.pwm.pwm_start,args=(self.q,))
        self.p.start()

    def car_move_forward(self):
        GPIO.output(self.backMotorInput2, GPIO.HIGH)
        GPIO.output(self.backMotorInput1, GPIO.LOW)
        self.car_turn_straight()

    def car_move_backward(self):
        GPIO.output(self.backMotorInput2, GPIO.LOW)
        GPIO.output(self.backMotorInput1, GPIO.HIGH)
        self.car_turn_straight()

    def car_turn_left(self):
        GPIO.output(self.frontMotorInput1, GPIO.LOW)
        GPIO.output(self.frontMotorInput2, GPIO.HIGH)
        GPIO.output(self.frontMotorEn, GPIO.HIGH)

    def car_turn_right(self):
        GPIO.output(self.frontMotorInput1, GPIO.HIGH)
        GPIO.output(self.frontMotorInput2, GPIO.LOW)
        GPIO.output(self.frontMotorEn, GPIO.HIGH)

    def car_stop(self):
        GPIO.output(self.backMotorInput1, GPIO.LOW)
        GPIO.output(self.backMotorInput2, GPIO.LOW)

    def car_turn_straight(self):
        GPIO.output(self.frontMotorEn, GPIO.LOW)

    def clean_GPIO(self):
        self.sig.run_flag = False
        self.q.put(self.sig)
        GPIO.cleanup()

if __name__ == '__main__':
    speed = 0.5 # parameter of speed: 0~1, 1 represents high speed
    cc = car_control(speed)

    cc.car_move_forward()
    time.sleep(2)

    cc.clean_GPIO()

