# 相机标定：从参数到可用的坐标

标定的目的不是"跑通 `calibrateCamera`"，而是让**像素坐标到物理坐标的映射误差**满足你的精度要求。先想清楚精度目标，再决定标定板、镜头和流程。

## 1. 模型与参数

### 针孔模型

```
[ u ]   [ fx  0  cx ] [ X ]
[ v ] = [ 0  fy  cy ] [ Y ]
[ 1 ]   [ 0   0   1 ] [ Z ]
```

内参 5 个：`fx, fy, cx, cy`（+ 畸变系数）。外参：旋转 `R`、平移 `t`。

- `fx = f / dx`，`dx` 是像元物理尺寸。**像素焦距不等于物理焦距**，别混用。
- `cx, cy` 理论上是图像中心，实际常在中心附近偏几个像素，以标定结果为准。

### 畸变模型

OpenCV 默认 `k1, k2, p1, p2, k3`：

```
x' = x(1 + k1 r² + k2 r⁴ + k3 r⁶) + 2 p1 x y + p2(r² + 2x²)
y' = y(1 + k1 r² + k2 r⁴ + k3 r⁶) + p1(r² + 2y²) + 2 p2 x y
```

- 普通工业镜头：5 参数够用。
- 广角/鱼眼：必须用 `fisheye` 模型（`cv2.fisheye.*`），否则边缘残差会大到不可用。
- 畸变系数对噪声很敏感：标定板覆盖不到的画面边缘，系数会外推得很野。**标定板要覆盖实际使用的整个视场**，尤其是四角。

## 2. 张正友标定实操

### 采集

- 标定板占画面 1/3 ~ 1/2，在视场里**均匀分布**：中心、四角、边缘都要有。
- 姿态多样：至少 3 个明显不同的倾角（建议 15°~45°），只平移不旋转会退化。
- 15~25 张足够；张数不是越多越好，**覆盖度**才是。
- 板面必须平整，打印后贴在硬板上；用屏幕显示标定板会因摩尔纹和屏幕平整度引入误差。
- 固定焦距、关闭自动对焦和自动曝光后再拍。

### 求解

```python
import cv2, numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
ok, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, image_size, None, None,
    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    criteria=criteria,
)
```

角点检测用 `findChessboardCornersSB`（OpenCV ≥ 4.5）比老的 `findChessboardCorners` 稳定得多，且自带亚像素。

### 评估

```python
total_err = 0
for i in range(len(object_points)):
    proj, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], K, dist)
    err = cv2.norm(image_points[i], proj, cv2.NORM_L2) / len(proj)
    total_err += err
print("reprojection error (px):", total_err / len(object_points))
```

- 重投影误差 **< 0.5 px**：一般可用；**< 0.2 px**：较好。
- 但重投影误差低 **不等于** 物理精度高：它只说明模型自洽。真正的验证是**用标定结果量一个已知尺寸**（如量块、棋盘格边长），看相对误差。
- 逐张误差列表比总平均更有用：某一张误差 3 px，通常是那张角点检测错了，直接剔除重标。

## 3. 手眼标定

### 两种构型

| 构型 | 相机装在哪 | 求什么 | OpenCV |
| :--- | :--- | :--- | :--- |
| eye-in-hand | 机械臂末端 | 相机→末端 | `CALIB_HAND_EYE_TSAI` 等 |
| eye-to-hand | 固定支架 | 相机→基座 | 需先转成 eye-in-hand 形式 |

`cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)`，输入至少 10 组以上、姿态差异明显的位姿。

### 关键前提

- 标定板固定在末端（eye-in-hand）或固定在工作台（eye-to-hand），**采集期间不能动**。
- 机械臂位姿从控制器读，注意**单位**（mm vs m）和**坐标系定义**（欧拉角顺序、四元数顺序）。
- 旋转要足够丰富：只绕一个轴转，旋转部分会病态。

### 验证

标定完必须验证：让机械臂走几个**未参与标定**的点，用标定结果算物理坐标，与真实坐标比较。误差应在你的工艺容差内（通常 0.2~1 mm 量级，取决于臂和相机）。

## 4. 常见坑

1. **标定板和实际工作距离不一致**：标定距离要覆盖实际景深，否则 `fx` 偏差会直接变成尺度误差。
2. **换了镜头/调了焦距没重标**：任何光学变化都必须重标。
3. **图像被缩放后再标定**：`fx, cx` 会跟着缩放，混用不同分辨率的图必然错。
4. **畸变校正后忘了改内参**：`undistort` 后仍是同一套 `K`，但如果先 resize 再校正，参数要同步缩放。
5. **把像素误差当物理误差**：`0.3 px` 在 5 m 工作距离、大视场下可能对应几毫米。
6. **标定板不平**：打印纸贴不牢会引入系统性误差，且重投影误差看不出来。

## 5. 检查清单

- [ ] 标定板覆盖全视场（含四角），姿态 ≥ 3 种倾角
- [ ] 采集时焦距、曝光、增益锁定
- [ ] 逐张重投影误差已检查，异常帧已剔除
- [ ] 用已知尺寸物体验证物理精度，而不只看重投影误差
- [ ] 手眼标定用未参与标定的点位做独立验证
- [ ] 标定结果与镜头、分辨率、工作距离一起记录归档
