## opencv 常见命令

`cv2.imread(path, falg)` 

path - ⽂件存放的本地路径； 

flag - 指定⽂件读取的具体⽅式； 

flag有以下三种不同的⽅式： 

cv2.IMREAD_COLOR，以RGB形式读取数据； 

cv2.IMREAD_GRAYSCALE，以灰度图形式读取数据； 

cv2.IMREAD_UNCHANGED，以原始数据格式读取数据。

`cv2.imshow(window_name, image)` 

该函数讲在指定的window_name窗⼝中展示指定的图像内容Image.

`cv2.write(path, image)` 

是通过制定存放路径path和存放图像image将图像保存在制定的本地路径处. 

`cv2.waitKey(time)` 

该函数为按键等待函数，等待时间为⽤户⾃⾏指定的时间time，单位为微秒。如果时间time被设置为0，则该函数为⼀直等待直到⽤户触发关闭图⽚。

> 灰度图的计算公式为：0.2989*R+0.5870*G+0.1140*B
>
> rgb在第三个维度分别表示为[:, :, 2]\[:, :, 1]\[:, :, 0]，α在第四维
>
> *RGB**通道图像，在**OpenCV**中是以**BGR形式进⾏保存的*
>
> *#* *即* *channel_0 -* *蓝⾊，**channel_1 -* *绿⾊，* *channel_2 -* *红⾊*

`rows, cols, channels = img.shape  *获取图像的⾏、列与通道*`

`cv2.resize(img, tagrt_img_size, interpolation = cv2.INTER_LINEAR)`插值

`tagrt_img_size = (1000, 1000)`

- 双线性插值 - cv2.INTER_LINEAR 
- 最近邻域插值 - cv2.INTER_NEAREST； 
- 4x4像素邻域的双三次插值 - cv2.INTER_CUBIC； 
- 8*8像素邻域的Lanczos插值 - cv2.INTER_LANCZOS4； 
- 区域插值 - cv2.INTER_AREA； 

`cv2.resize(src, dsize, fx, fy, interpolation)`

- src - 待缩放图像； 
- dsize - 输出图像所需⼤⼩； 
- fx - 沿⽔平轴的⽐例因⼦（可选）；
- fy - 沿垂直轴的⽐例因⼦（可选）；
- interpolation - 差值⽅式（可选），共有以上介绍5种⽅式；

`cv2.getRotationMatrix2D(center,degrees,scale)` 

- 该函数为⽣成旋转矩阵 
- center - 旋转中⼼点，如果是中⼼旋转就是(cols/2,rows/2)； 
- degrees - 旋转⻆度，案例中为45度；
- scale - 缩放尺度，控制⽣成结果⼤⼩，案例中为1，不进⾏缩放；

`cv2.warpAffine(img,rot_mat,(cols,rows))` 

- 该函数为对原始图像进⾏仿射变换 
- img - 原始输⼊图像； 
- rot_mat - 仿射变化所需要的旋转矩阵； 
- (cols, rows) - 变换后的图像尺⼨，注意是**cols**在前，**rows**在后；

`cv2.flip(img,flipcode)` 

- img - 原始输⼊图像； 
- flipcode - 翻转⽅式，1为⽔平翻转；0为垂直翻转；-1为⽔平垂直翻转； 

`cv2.GaussianBlur(src,ksize, sigmaX, sigmaY, borderType)`

- src - 原始输⼊图像； 
- ksize - ⾼斯核⼤⼩，必须为正数和奇数；
- sigmaX - X⽅向上的⾼斯核标准偏差；
- sigmaY - Y⽅向上的⾼斯核标准差； 
- borderType - 像素外推的⽅法，具体可以参考OpenCV官⽅解释；

`cv2.rectangle(img, pt1, pt2, color, thickness, lineType, shift)` 

- 该函数通过指定左上，右下两个顶点以及线条颜⾊和线条绘制等内容，⾃动绘制矩形框； 
- img - 输⼊图像； 
- pt1, pt2 - 分别代指左上和右下两个坐标点； 
- color - 线条绘制的颜⾊； 
- thickness - 线条绘制的粗细，thickness=1则线条粗细为⼀个像素，如果为-1则是将矩形框使⽤指定颜⾊进⾏填充；
- lineType - 线条类型，⼀般不⽤特别指定；
- shift - ⽤于处理坐标中的⼩数位数，⼀般不⽤特别指定； 