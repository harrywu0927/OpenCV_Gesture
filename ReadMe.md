### 使用方法

(需要先安装OpenCV库)

```
cmake .
make
```

#### 录制手势
运行 writeFeatures 程序将拍摄若干图片并提取特征，写入文件

#### 识别
运行 main 摆出手势

### 实现目标
- 调取摄像头，逐帧识别手部
- 分类出手势
- 逐帧显示结果

### 效果展示
![](images/Screenshot%202023-10-22%20at%2020.16.42.png)


整体流程如下：
![](images/Screenshot%202023-10-22%20at%2020.03.55.png)

#### 获取皮肤部分——YCrCb颜色空间Cr分量+OTSU法阈值分割
- YCrCb颜色空间的Cr分量反映了输入信号红色部分与RGB信号亮度值之间的差异。
该方法的原理：
- 将RGB图像转换到YCrCb颜色空间，提取Cr分量（红色分量）图像
- 对Cr分量做自适应二值化阈值分割处理（OTSU法），OTSU算法可以对前景和后景进行区分

#### OTSU法
- OTSU是一种确定图像二值化分割阈值的算法，从原理上讲又称作最大类间方差法。
- 被认为是图像分割中阈值选取的最佳算法，计算简单，不受图像亮度和对比度的影响。
- 能够把图像分成背景和前景两部分，因为方差是灰度分布均匀性的一种度量,背景和前景之间的类间方差越大,说明构成图像的两部分的差别越大。
---- Otsu N. A threshold selection method from gray-level histograms[J]. IEEE transactions on systems, man, and cybernetics, 1979, 9(1): 62-66.
![](images/Screenshot%202023-10-22%20at%2020.24.53.png)

#### OpenCV 级联分类器
使用OpenCV自带的面部级联分类器可以方便地获取人的脸部。

``` cpp
CascadeClassifier face_cascade;
String face_cascade_name = samples::findFile(“data/haarcascades/haarcascade_frontalface_alt.xml");
face_cascade.load(face_cascade_name)

Mat frame_gray;
cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
equalizeHist(frame_gray, frame_gray);

//-- Detect faces
std::vector<Rect> faces;
face_cascade.detectMultiScale(frame_gray, faces);
for (size_t i = 0; i < faces.size(); i++)
{
    rectangle(frame, faces[i], cv::Scalar(255, 0, 0), 1);
}
```

#### 寻找物体轮廓
OpenCV同样为我们提供了获取物体轮廓的便利方法，findContours函数将物体的封闭轮廓提取为一组点集存放在数组中。选择图像中轮廓最大的物体视为手部。

```cpp
Mat ImageBinary = ImageIn;
vector<vector<Point>> contours; //定义轮廓向量
findContours(ImageBinary, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE); 
//寻找轮廓
```

#### 获取图像特征
- 在本项目中，我们使用傅立叶描绘子来表示手部轮廓的特征。
- 傅立叶描绘子是一种边界描绘子，对于旋转以及缩放不十分敏感。
- 由于形状的能量大多集中在低频部分，高频部分一般很小且容易受到干扰，因此只取前12位。
![](images/Screenshot%202023-10-22%20at%2020.28.35.png)
- 这样计算出的傅立叶级数与形状尺度、方向和曲线起始点S0都有关系，因此以a(1)为基准归一化:
![](images/Screenshot%202023-10-22%20at%2020.28.42.png)


#### 形态学处理
- 使用图像分割二值化得到的手部边缘有非常多的毛刺，这将会使傅立叶变换的结果变得非常不稳定。
- 腐蚀：去除图像中不想要的小细节，比如一张二值图片中的噪点或者小细节。
- 膨胀：放大细节。
- 先腐蚀后膨胀：去除孤立的小点，毛刺
- 先膨胀后腐蚀：填平小孔，弥合小裂缝

```cpp
Mat element = getStructuringElement(MORPH_ELLIPSE, Size(25, 25));
Mat eroded, out;
erode(Binary, eroded, element);
dilate(eroded, out, element);
```

![](images/Screenshot%202023-10-22%20at%2020.31.59.png)

#### 训练分类模型
- 获取到手部特征后（傅立叶级数的前12个系数），可以将其视为一个12维向量，映射到12维空间中的一个点。
- 同一个手势的特征点一定在12维空间中簇集在一起，因此可以使用KNN算法对手势分类。
- 对于数据集的获取，使用同一手势对着摄像头录制手部轮廓特征，适当旋转、平移，每个手势持续400帧，将每帧计算得到的前12个傅立叶级数的系数存为二进制文件，五个手势共5个.fea文件
下面是KNN算法的图形化表示：
![](images/Screenshot%202023-10-22%20at%2020.33.10.png)

- 使用OpenCV提供的机器学习模块中的KNN算法。
- 读取录制好的样本数据到trainData，设置标签trainLabels
- 设置合适的K值后传入对应参数就可以开始训练
```cpp
int K = 20;
cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
knn->setDefaultK(K);
knn->setIsClassifier(true);
knn->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
knn->train(trainData, ml::ROW_SAMPLE, trainLabels);
```

#### 已知缺陷
- 对于肤色的识别高度依赖颜色空间中的红色分量，因此若场景中混杂了较多的木制或木色系家具，或穿着偏红色系的衣服将影响识别效果。
- 手部轮廓识别时按照轮廓最大的物体判断，当场景出现较大的干扰色物体时，将可能识别不到手部。一种可能的解决方案是使用异常检测的手段，将离手部特征点较远的杂物的特征点视为离群点，检测到离群点后继续获取图像中第二大的轮廓，再将其视为手部。