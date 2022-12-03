###  Camera类
#### API
```python
class Camera(object):
    def __init__(self,resolution,framerate,vflip,hflip):
        # 相机初始化分辨率
        self.resolution = resolution
        # 相机初始化帧率
        self.framerate = framerate
        # 相机画面是否垂直翻转，布尔值
        self.vflip = vflip
        # 相机画面是否水平翻转，布尔值
        self.hflip = hflip

        """
            其他相机初始化相关代码
        """

    def start_recording(self,processor):
        # processor为用户传入的帧处理类
        self.processor = processor

        """
            其他录制初始化相关代码
        """

    def wait_recording(self,time):
        # time为用户定义的视频录制时间
        # 对于录制时长内的每一帧，frame
        self.processor.write(frame)

        """
            其他帧处理相关代码
        """

    def stop_recording(self):
        self.processor.flush()

        """
            其他录制结束相关代码，例如：
            cap.release()
            cv.destroyAllWindows()
        """
```

### 帧处理类
```python
class ProcessOutput(object):
    def __init__(self):
        # 例如统计处理帧总数量
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
```

### 用户使用示例
```python
    camera = Camera(resolution=(160, 120), framerate=30, vflip=True, hflip=False)
    poutput = ProcessOutput()
    camera.start_recording(processor=poutput)
    camera.wait_recording(time=120)
    camera.stop_recording()
```

### 更新说明
#### collect_data分支
##### 第一次更新
- 增加`collect_data.py`，用于收集训练数据
- `Class Processor`，帧处理类放置于`collect_data.py`文件中，更方便的自定义 `write()`函数行为，以及接收全局变量`key`
	- 在正式的自动驾驶代码中，`write()`函数需要重新定义，以通过卷积神经网络
	- `write()`函数中的文件保存名字加入了按键`key`，即图像的标签
- `Camera.start_recording()`，帧处理类的实例化交给用户，例如`collect_data.py`第43-44行用于查看帧处理实例的相关信息
- `Camera.wait_recording()`，更改为记录固定时长，并删除`waitKey()`