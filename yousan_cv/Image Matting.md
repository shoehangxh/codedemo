## Image Matting

### 4通道图像

- RGBA

- 软分割（不是每一个像素一个分类，而是一个分类的概率），抠图

- 基于trimap的matting框架

  trimap：三值图：确定性前景r，确定性背景b，不确定区域g

- 任意切换背景

- matting基准

### 基于trimap的模型

- deep image matting
  - trimap作为监督输入
  - alpha matte损失
  - RGB