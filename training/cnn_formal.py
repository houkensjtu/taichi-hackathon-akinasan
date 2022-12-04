import taichi as ti
# 注意单浮点精度问题
import glob
import numpy as np
import math
import random

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

def cnntrain(cnn,inputdata,outputdata,opts,train_num):
    cnn.L = ti.field(ti.float64,shape=(train_num,))
    input_field = ti.field(dtype=ti.f64,shape=(1,28,28))
    output_field = ti.field(dtype=ti.f64,shape=(5,))
    for e in range(opts.numepochs):
        for n in range(train_num):
            opts.alpha = 0.03 - 0.029 * n / (train_num - 1)

            #m = random.randint(0,219) # a=<n<=b,随机抽样

            generate_input_field(input_field,inputdata[n,:,:])
            output_field.from_numpy(outputdata[n,:])

            cnnff(cnn,input_field)
            print('cnn.OutLayer.y', cnn.OutLayer.y)
            cnnbp(cnn,output_field)
            cnnapplygrads(cnn,opts,input_field)

            l = 0.0
            for i in range(cnn.OutLayer.outputNum):
                l = l-output_field[i]*math.log(cnn.OutLayer.y[i]+1e-10)
            cnn.L[n] = l

            cnnclear(cnn)

            print("n={},f={},alpha={}".format(n,cnn.L[n],opts.alpha))

@ti.kernel
def generate_input_field(tf:ti.template(),arr:ti.types.ndarray()):
    for i,j in ti.ndrange(28,28):
        tf[0,i,j] = arr[i,j]

def load_data():
    image_array = np.zeros((1, 28, 28))
    label_array = np.zeros((1, 5), 'float')

    training_data = glob.glob('training_data_npz/*.npz')
    if not training_data:
        print("No training data in directory, exit")
        sys.exit()
    for single_npz in training_data:
        print(single_npz)
        if single_npz == 'training_data_npz\\1670119618.npz':
            with np.load(single_npz) as data:
                train_temp = data['train_imgs']
                train_labels_temp = data['train_labels']
            image_array = np.vstack((image_array, train_temp))
            label_array = np.vstack((label_array, train_labels_temp))

    return (image_array[1:, :],label_array[1:, :])

def save_cnn(cnn,path):
    np.save(path+'/clayer1_map.npy',cnn.CovLayer_1.mapData.to_numpy())
    np.save(path+'/layer1_bias.npy',cnn.CovLayer_1.biasData.to_numpy())

    np.save(path+'/clayer2_map.npy',cnn.CovLayer_2.mapData.to_numpy())
    np.save(path+'/clayer2_bias.npy',cnn.CovLayer_2.biasData.to_numpy())

    np.save(path+'/outlayer_w.npy',cnn.OutLayer.wData.to_numpy())
    np.save(path+'/outlayer_bias.npy',cnn.OutLayer.biasData.to_numpy())

def load_cnn(cnn,path):
    cnn.CovLayer_1.mapData.from_numpy(np.load(path+'/clayer1_map.npy'))
    cnn.CovLayer_1.biasData.from_numpy(np.load(path+'/layer1_bias.npy'))

    cnn.CovLayer_2.mapData.from_numpy(np.load(path+'/clayer2_map.npy'))
    cnn.CovLayer_2.biasData.from_numpy(np.load(path+'/clayer2_bias.npy'))

    cnn.OutLayer.wData.from_numpy(np.load(path+'/outlayer_w.npy'))
    cnn.OutLayer.biasData.from_numpy(np.load(path+'/outlayer_bias.npy'))
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

def cnntest(cnn,inputdata,outputdata,tst_num):
    print("size of test set:{}".format(tst_num))
    incorrectnum = 0
    tinput_field = ti.field(dtype=ti.f64,shape=(1,28,28))
    toutput_field = ti.field(dtype=ti.f64,shape=(5,))
    for n in range(tst_num):
        generate_input_field(tinput_field, inputdata[n, :, :])
        toutput_field.from_numpy(outputdata[n, :])
        cnnff(cnn,tinput_field)
        y_max_index = max_index(cnn.OutLayer.y)
        tag_max_index = max_index(toutput_field)

        if y_max_index != tag_max_index:
            incorrectnum += 1
            print("n:{},识别失败".format(n))
        else:
            print("n:{},识别成功".format(n))
        cnnclear(cnn)
    print("incorrect num:{}".format(incorrectnum))
    return incorrectnum/tst_num

if __name__ == '__main__':
    train_images, train_labels = load_data()
    train_images = train_images/255.0
    print(train_images.shape,train_labels.shape)
    print(train_labels)

    cnn = cnn_network(28,28) 
    opts = train_opts()
    opts.numepochs = 20
    opts.alpha = 0.03

    path = 'trained_model_office'
    cnntrain(cnn, train_images, train_labels, opts, 525) 
    save_cnn(cnn,path)

    unsuccess = cnntest(cnn,train_images,train_labels,525) #training set is used as test set
    print("成功率：{}".format((1-unsuccess)*100))