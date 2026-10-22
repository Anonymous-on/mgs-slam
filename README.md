# MGS-SLAM

面向单目 RGB 视频的高斯 SLAM 系统。

## 论文方案

输入连续图像后，先预测稠密点图和置信度，通过跨帧点图匹配估计相机位姿，并在 Sim(3) 位姿图中统一尺度。跟踪时估计动态区域的不确定性，降低运动物体对位姿优化的干扰。建图时依据颜色和深度变化自适应选点，初始化 3D 高斯；随后渲染颜色与空间变化的深度，联合优化高斯地图和相机位姿。

`单目 RGB → 点图匹配与位姿优化 → 高斯初始化 → 颜色/深度渲染 → 联合优化`

## 运行入口

```bash
python run.py --dataset /path/to/rgb_frames --mast3r-checkpoint /path/to/checkpoint.pth --output /path/to/output
```
