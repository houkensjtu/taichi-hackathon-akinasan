import taichi as ti
# 注意单浮点精度问题
import glob
import numpy as np
import math
import random
from ctlPWM import car_control
from Camera import Camera
import io
import threading
import time

#ti.init(arch=ti.gpu,device_memory_GB=4.0,default_fp=ti.f64)
ti.init(arch=ti.cpu,default_fp=ti.f64)

# define of convolution layer
@ti.data_oriented
class convolution_layer:
    def __init__(self,width,height,mapsize,inchannels,outchannels,isfullconnect=True):
        self.inputWidth = int(width)
        self.inputHeight = int(height)
        self.mapSize = mapsize
        self.inChannels = inchannels
        self.outChannels = outchannels
        self.isFullConnect = isfullconnect

        # valid mode
        self.outW = int(self.inputWidth - self.mapSize+1)
        self.outH = int(self.inputHeight - self.mapSize + 1)

        self.mapData = ti.field(dtype=ti.f64,shape=(inchannels,outchannels,mapsize,mapsize))
        self.biasData = ti.field(dtype=ti.f64,shape=(outchannels,)) # bias


        self.v = ti.field(dtype=ti.f64,shape=(outchannels,self.outH,self.outW))
        self.y = ti.field(dtype=ti.f64,shape=(outchannels,self.outH,self.outW))
        self.d = ti.field(dtype=ti.f64,shape=(outchannels,self.outH,self.outW))

        self.coor = ti.field(dtype=ti.f64,shape=(inchannels,self.inputHeight,self.inputWidth))
        self.flipmap = ti.field(dtype=ti.f64,shape=(inchannels,self.mapSize,self.mapSize))
        self.fill_d = ti.field(dtype=ti.f64,shape=(outchannels,self.outH+2*self.mapSize-2,self.outW+2*self.mapSize-2))
        self.mat_field = ti.Matrix.field(self.mapSize,self.mapSize,ti.f64,shape=(inchannels,3))

    @ti.kernel
    def initialize(self):
        for i,j,r,c in ti.ndrange(self.inChannels,self.outChannels,self.mapSize,self.mapSize):
            randnum = (ti.random()-0.5)*2
            self.mapData[i,j,r,c] = randnum *ti.sqrt(6.0/(self.mapSize*self.mapSize*(self.inChannels+self.outChannels)))

        self.biasData.fill(0.)
        self.v.fill(0.)
        self.y.fill(0.)
        self.d.fill(0.)

        mid_ind = int(self.mapSize/2)
        for i,j in self.mat_field:
            if j == 0:
                for x,y in ti.static(ti.ndrange(self.mapSize,self.mapSize)):
                    if x+y == mid_ind*2:
                        self.mat_field[i,0][x,y] = 1.0


#池化层
@ti.data_oriented
class polling_layer:
    def __init__(self,width,height,mapsize,inchannels,outchannels,pooltype):
        self.inputWidth = int(width)
        self.inputHeight = int(height)
        self.mapSize = mapsize
        self.inChannels = inchannels
        self.outChannels = outchannels
        self.poolType = pooltype

        self.outW = int(self.inputWidth/self.mapSize)
        self.outH = int(self.inputHeight/self.mapSize)

        self.biasData = ti.field(dtype=ti.f64, shape=(outchannels,))
        self.y = ti.field(dtype=ti.f64,shape=(outchannels,self.outH,self.outW))
        self.d = ti.field(dtype=ti.f64,shape=(outchannels,self.outH,self.outW))
        self.max_position = ti.field(dtype=ti.i32,shape=(outchannels,self.outH,self.outW))



    @ti.kernel
    def initialize(self):
        self.biasData.fill(0.)
        self.y.fill(0.)
        self.d.fill(0.)
        self.max_position.fill(0)

#输出层
@ti.data_oriented
class out_layer:
    def __init__(self,inputNum,outputNum,isfullconnect=True):
        inputNum = int(inputNum)
        outputNum = int(outputNum)
        self.inputNum = inputNum
        self.outputNum = outputNum
        self.isFullConnect = isfullconnect

        self.wData = ti.field(dtype=ti.f64,shape=(outputNum,inputNum))
        self.biasData = ti.field(dtype=ti.f64, shape=(outputNum,))

        self.v = ti.field(dtype=ti.f64, shape=(outputNum,))
        self.y = ti.field(dtype=ti.f64, shape=(outputNum,))
        self.d = ti.field(dtype=ti.f64, shape=(outputNum,))

    @ti.kernel
    def initialize(self):
        self.biasData.fill(0.)
        self.v.fill(0.)
        self.y.fill(0.)
        self.d.fill(0.)
        for i,j in ti.ndrange(self.outputNum,self.inputNum):
            randnum = (ti.random()-0.5)*2
            self.wData[i,j] = randnum*ti.sqrt(6.0/(self.inputNum+self.outputNum))

# cnn of 5 layer
@ti.data_oriented
class cnn_network:
    def __init__(self,init_h,init_w):
        self.layerNum = 5
        self.CovLayer_1 = convolution_layer(height=init_h,width=init_w,mapsize=5,inchannels=1,outchannels=6)
        self.CovLayer_1.initialize()
        pl1_h = init_h-self.CovLayer_1.mapSize+1
        pl1_w = init_w - self.CovLayer_1.mapSize + 1

        self.PoolLayer_1 = polling_layer(height=pl1_h,width=pl1_w,mapsize=2,inchannels=self.CovLayer_1.outChannels,outchannels=self.CovLayer_1.outChannels,pooltype=1)
        self.PoolLayer_1.initialize()
        cl2_h = pl1_h/self.PoolLayer_1.mapSize
        cl2_w = pl1_w / self.PoolLayer_1.mapSize

        self.CovLayer_2 = convolution_layer(height=cl2_h,width=cl2_w,mapsize=5,inchannels=self.PoolLayer_1.outChannels,outchannels=12)
        self.CovLayer_2.initialize()
        pl2_h = cl2_h-self.CovLayer_2.mapSize+1
        pl2_w = cl2_w-self.CovLayer_2.mapSize+1

        self.PoolLayer_2 = polling_layer(height=pl2_h,width=pl2_w,mapsize=2,inchannels=self.CovLayer_2.outChannels,outchannels=self.CovLayer_2.outChannels,pooltype=1)
        self.PoolLayer_2.initialize()
        ol_h = pl2_h/self.PoolLayer_2.mapSize
        ol_w = pl2_w/self.PoolLayer_2.mapSize

        self.OutLayer = out_layer(inputNum=ol_h*ol_w*self.PoolLayer_2.outChannels,outputNum=5)
        #self.OutLayer = out_layer(inputNum=ol_h * ol_w * self.PoolLayer_2.outChannels, outputNum=3)
        self.OutLayer.initialize()
        print(self.OutLayer.inputNum,self.OutLayer.outputNum)

        self.e = ti.field(ti.f64,shape=(self.OutLayer.outputNum,))
        self.e.fill(0.)
        self.L = None

# 训练参数
@ti.data_oriented
class train_opts:
    def __init__(self):
        self.numepochs = None
        self.alpha = None

# 卷积
@ti.func
def cov_multiply(i,j,inpudata,mapdata,oh,ow,mapsize):
    s = 0.0
    for x,y in ti.ndrange(mapsize,mapsize):
        s += inpudata[j,oh+x,ow+y]*mapdata[j,i,x,y]
    return s

#Relu
@ti.func
def activation_sigma(inputd,bias):
    temp = inputd+bias
    if temp <= 0:
        temp = 0
    return temp

@ti.kernel
def cov_layer_ff(inputdata:ti.template(),covlayer:ti.template()):
    num,rows,cols = inputdata.shape

    total_num = rows*cols
    for i in range(num):
        sum_temp = 0.0
        covriance = 0.0
        for j,k in ti.ndrange(rows,cols):
            sum_temp += inputdata[i,j,k]
        mean = sum_temp/total_num
        for j,k in ti.ndrange(rows,cols):
            covriance += (inputdata[i, j, k] - mean) ** 2
        std = ti.sqrt(covriance / total_num)
        for j,k in ti.ndrange(rows,cols):
            inputdata[i, j, k] = (inputdata[i, j, k] - mean) / (std+0.01)

    for i,j in ti.ndrange(covlayer.outChannels,covlayer.inChannels):
        for oh,ow in ti.ndrange(covlayer.outH,covlayer.outW):
            covlayer.v[i,oh,ow] += cov_multiply(i,j,inputdata,covlayer.mapData,oh,ow,covlayer.mapSize)


    for i,r,c in ti.ndrange(covlayer.outChannels,covlayer.outH,covlayer.outW):
        covlayer.y[i,r,c] = activation_sigma(covlayer.v[i,r,c],covlayer.biasData[i])

@ti.kernel
def pool_layer_ff(inputdata:ti.template(),poolayer:ti.template()):
    mpszie = poolayer.mapSize
    num,rows,cols = inputdata.shape
    for i,r,c in ti.ndrange(poolayer.outChannels,poolayer.outH,poolayer.outW):
        max = -99999999.0
        max_index = 0
        for m,n in ti.ndrange((r*mpszie,r*mpszie+mpszie),(c*mpszie,c*mpszie+mpszie)):
            if inputdata[i,m,n] > max:
                max = inputdata[i,m,n]
                max_index = m*cols+n
        poolayer.y[i,r,c] = max
        poolayer.max_position[i,r,c] = max_index

@ti.func
def handle_nan(num):
    temp = 0.1
    if not ti.math.isnan(num):
        temp = num
    return temp

@ti.kernel
def outlayer_ff(inputdata:ti.template(),outlayer:ti.template()):
    num,rows,cols = inputdata.shape
    for i in range(outlayer.outputNum):
        s = 0.0
        for j,r,c in ti.ndrange(num,rows,cols):
            k = j*rows*cols+r*cols+c
            s += inputdata[j,r,c]*outlayer.wData[i,k]
        outlayer.v[i] = s

    #softmax
    sum = 0.0
    for i in range(outlayer.outputNum):
        yi = ti.exp(outlayer.v[i]+outlayer.biasData[i])
        sum += yi
        outlayer.y[i] = yi

    for i in range(outlayer.outputNum):
        outlayer.y[i] = outlayer.y[i]/sum


#前向传播
def cnnff(cnn,inputdata):
    cov_layer_ff(inputdata,cnn.CovLayer_1)

    pool_layer_ff(cnn.CovLayer_1.y,cnn.PoolLayer_1)

    cov_layer_ff(cnn.PoolLayer_1.y,cnn.CovLayer_2)

    pool_layer_ff(cnn.CovLayer_2.y,cnn.PoolLayer_2)

    outlayer_ff(cnn.PoolLayer_2.y,cnn.OutLayer)


#反向传播
# softmax->affine
@ti.kernel
def softmax_bp(outputdata:ti.template(),e:ti.template(),o:ti.template()):
    for i in range(o.outputNum):
        e[i] = o.y[i]-outputdata[i]
        o.d[i] = e[i]*sigma_derivation(o.y[i])


# affine->s4
@ti.kernel
def full2pool_bp(o:ti.template(),s:ti.template()):
    oh = s.outH
    ow = s.outW
    for i in range(s.outChannels):
        for r in range(oh):
            for c in range(ow):
                wInd = i*oh*ow + r*ow +c
                for j in range(o.outputNum):
                    s.d[i,r,c] = s.d[i,r,c] + o.d[j]*o.wData[j,wInd]

@ti.func
def maxUpSample(S,i,r,c):
    temp = 0.0
    num,rows,cols = S.d.shape
    mpsize = S.mapSize

    out_r = rows*mpsize
    out_c = cols*mpsize
    for j,k in ti.ndrange(rows,cols):
        index_r = int(S.max_position[i, j, k] / out_c)
        index_c = int(S.max_position[i, j, k] % out_c)
        if index_r == r and index_c == c:
            temp = S.d[i, j, k]
    return temp


@ti.func
def sigma_derivation(num):
    temp = 0
    if num>0:
        temp = 1
    return temp

@ti.kernel
def pool2cov_bp(S:ti.template(),C:ti.template()):
    num,rows,cols = C.d.shape

    for i in range(C.outChannels):
        for r,c in ti.ndrange(rows,cols):
            C.d[i, r, c] = maxUpSample(S,i,r,c) * sigma_derivation(C.y[i, r, c])



@ti.func
def flip_kernel(C,i,j):
    for x in ti.static(range(C.mapSize)):
        for y in ti.static(range(C.mapSize)):
            C.mat_field[i,1][x,y] = C.mapData[i,j,x,y]

    C.mat_field[i,2] = C.mat_field[i,0]@C.mat_field[i,1]@C.mat_field[i,0]
    for x in ti.static(range(C.mapSize)):
        for y in ti.static(range(C.mapSize)):
            C.flipmap[i,x,y] = C.mat_field[i,2][x,y]


@ti.func
def cov(C,i,j):
    mpsize = C.mapSize

    inchannels,coor_r,coor_c = C.coor.shape

    flip_kernel(C,i,j)

    for x,y in ti.ndrange(coor_r,coor_c):
        temp = 0.0
        for k,l in ti.ndrange(mpsize,mpsize):
            temp += C.fill_d[j,x+k,y+l]*C.flipmap[i,k,l]
        C.coor[i,x,y] = temp


@ti.func
def fill_d(C):
    nums,rows,cols = C.d.shape
    mpsize = C.mapSize
    fill_start_r = mpsize-1
    fill_start_c = mpsize-1
    for i,j,k in ti.ndrange(nums,(fill_start_r,rows+fill_start_r),(fill_start_c,fill_start_c+cols)):
        C.fill_d[i,j,k] = C.d[i,j-fill_start_r,k-fill_start_c]

# C -> S
@ti.kernel
def cov2pool_bp(C:ti.template(),S:ti.template()):
    num,rows,cols = S.d.shape
    fill_d(C)
    for i in range(C.inChannels):
        for j in range(C.outChannels):
            cov(C,i,j)
            for x,y in ti.ndrange(rows,cols):
                S.d[i,x,y] = S.d[i,x,y] + C.coor[i,x,y]

def cnnbp(cnn,outputdata):
    softmax_bp(outputdata,cnn.e,cnn.OutLayer)
    full2pool_bp(cnn.OutLayer,cnn.PoolLayer_2)
    pool2cov_bp(cnn.PoolLayer_2,cnn.CovLayer_2)
    cov2pool_bp(cnn.CovLayer_2,cnn.PoolLayer_1)
    pool2cov_bp(cnn.PoolLayer_1,cnn.CovLayer_1)

@ti.kernel
def update_full_para(inputdata:ti.template(),opts:ti.template(),O:ti.template()):
    num,rows,cols = inputdata.shape
    mat_size = rows*cols

    for i in range(O.outputNum):
        for j in range(O.inputNum):
            x = int(j/mat_size)
            temp = int(j%mat_size)
            y = int(temp/cols)
            z = int(temp%cols)
            O.wData[i,j] = O.wData[i,j] - opts.alpha*O.d[i]*inputdata[x,y,z]

        O.biasData[i] =  O.biasData[i]-opts.alpha*O.d[i]

@ti.func
def cdk(inputdata,C,i,j,r,c):
    num, rows, cols = C.d.shape
    sum = 0.0
    for x,y in ti.ndrange(rows, cols):
        sum += C.d[i,x,y]*inputdata[j,r+x,c+y]
    return sum

@ti.kernel
def update_cov_para(inputdata:ti.template(),opts:ti.template(),C:ti.template()):
    num,rows,cols = C.d.shape
    for i in range(C.outChannels):
        for j in range(C.inChannels):
            for r,c in ti.ndrange(C.mapSize,C.mapSize):
                C.mapData[j,i,r,c] = C.mapData[j,i,r,c] -opts.alpha*cdk(inputdata,C,i,j,r,c)

        d_sum = 0.0
        for x,y in ti.ndrange(rows,cols):
            d_sum += C.d[i,x,y]
        C.biasData[i] = C.biasData[i] - opts.alpha*d_sum


def cnnapplygrads(cnn,opts,inputdata):
    update_cov_para(inputdata,opts,cnn.CovLayer_1) #C1
    update_cov_para(cnn.PoolLayer_1.y,opts,cnn.CovLayer_2) #C3
    update_full_para(cnn.PoolLayer_2.y,opts,cnn.OutLayer) #O5

@ti.kernel
def clear_cov_mid_para(C:ti.template()):
    num,rows,cols = C.d.shape
    for i in range(C.outChannels):
        for r,c in ti.ndrange(rows,cols):
            C.d[i,r,c] = 0.0
            C.v[i,r,c] = 0.0
            C.y[i,r,c] = 0.0

@ti.kernel
def clear_pool_mid_para(S:ti.template()):
    num,rows,cols = S.d.shape
    for i in range(S.outChannels):
        for r,c in ti.ndrange(rows,cols):
            S.d[i,r,c] = 0.0
            S.y[i,r,c] = 0.0

@ti.kernel
def clear_out_mid_para(O:ti.template()):
    for i in range(O.outputNum):
        O.d[i] = 0.0
        O.v[i] = 0.0
        O.y[i] = 0.0

def cnnclear(cnn):
    clear_cov_mid_para(cnn.CovLayer_1)
    clear_pool_mid_para(cnn.PoolLayer_1)
    clear_cov_mid_para(cnn.CovLayer_2)
    clear_pool_mid_para(cnn.PoolLayer_2)
    clear_out_mid_para(cnn.OutLayer)

@ti.kernel
def generate_input_field(tf:ti.template(),arr:ti.types.ndarray()):
    for i,j in ti.ndrange(28,28):
        tf[0,i,j] = arr[i,j]

def load_cnn(cnn):
    cnn.CovLayer_1.mapData.from_numpy(np.load('trained_model/clayer1_map.npy'))
    cnn.CovLayer_1.biasData.from_numpy(np.load('trained_model/layer1_bias.npy'))

    cnn.CovLayer_2.mapData.from_numpy(np.load('trained_model/clayer2_map.npy'))
    cnn.CovLayer_2.biasData.from_numpy(np.load('trained_model/clayer2_bias.npy'))

    cnn.OutLayer.wData.from_numpy(np.load('trained_model/outlayer_w.npy'))
    cnn.OutLayer.biasData.from_numpy(np.load('trained_model/outlayer_bias.npy'))

@ti.kernel
def max_index(f:ti.template()) -> int:
    lenth = f.shape[0]
    max_index = 0
    max_value = 0.0
    ti.loop_config(serialize=True)
    for i in range(lenth):
        if f[i] > max_value:
            max_value = f[i]
            max_index = i
    return max_index

def predict(buf):
    global latest_time, cc, cnn1
    generate_input_field(tinput_field, buf)
    cnnff(cnn1, tinput_field)
    y_max_index = max_index(cnn1.OutLayer.y)
    cnnclear(cnn1)
    return y_max_index

def car_move(value):
    global latest_time, cc, cnn1
    if action_num == 0:
        print("Left")
        cc.car_turn_left()
        time.sleep(0.25)
    elif action_num== 1:
        print("Right")
        cc.car_turn_right()
        time.sleep(0.25)
    elif action_num == 2:
        cc.car_move_forward()
        print('Forward')
    elif action_num == 3:
        cc.car_move_backward()
        print('Backward')
    else:
        cc.car_stop()
        print('Stop')

class ImageProcessor(threading.Thread):
    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.grayimg = None
        self.start()

    def run(self):
        global latest_time, cc, cnn1
        while not self.terminated:
            if self.event.wait(1):
                try:
                    current_time = time.time()
                    if current_time>latest_time:
                        if current_time-latest_time>1:
                            print("*" * 30)
                            print(current_time-latest_time)
                            print("*" * 30)
                        latest_time = current_time
                        pre_v = predict(self.grayimg)
                        car_move(pre_v)
                finally:
                    self.event.clear()
                    with self.owner.lock:
                        self.owner.pool.append(self)

def RGB2GRAY(img):
    temp_img = np.zeros([28,28])
    temp_img = img[:,:,0]*0.3+img[:,:,1]*0.59+img[:,:,2]*0.11
    return temp_img

class ProcessOutput(object):
    def __init__(self):
        self.done = False
        self.lock = threading.Lock()
        self.pool = [ImageProcessor(self) for i in range(4)]
        self.processor = None

    def write(self, buf):
        if self.processor:
            self.processor.event.set()
        with self.lock:
            if self.pool:
                self.processor = self.pool.pop()
            else:
                self.processor = None
        if self.processor:
            self.processor.grayimg = RGB2GRAY(buf)

    def flush(self):
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None
        while True:
            with self.lock:
                try:
                    proc = self.pool.pop()
                except IndexError:
                    pass 
            proc.terminated = True
            proc.join()

def main():
    global latest_time, cc, cnn1
    try:
        camera = Camera(resolution=(28, 28), framerate=30, vflip=False, hflip=False)
        outproc = ProcessOutput()

        camera.start_recording(outproc)
        camera.wait_recording(120)
        camera.stop_recording()
    finally:
        cc.clean_GPIO()

if __name__ == '__main__':
    global latest_time, cc, cnn1

    latest_time = time.time()

    tinput_field = ti.field(dtype=ti.f64,shape=(1,28,28))

    speed = 0.8
    cc = car_control(speed)

    cnn1 = cnn_network(28,28)
    load_cnn(cnn1)

    main()
