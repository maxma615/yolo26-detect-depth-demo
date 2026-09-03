# 【RDK S100P】YOLO26 检测 + 单目深度 双模型 BPU 并发实时演示（开源）

> 在 RDK S100P 上让 **YOLO26 目标检测** 和 **YOLO26 单目深度估计** 两个模型**同时**跑在单核 BPU 上，
> 一路合成画面同时输出到**浏览器（MJPEG）**和 **HDMI（DRM/KMS 直出）**，
> 1080p@30 零丢帧、BPU 占用 ~65%。代码已开源，欢迎拍砖 / 复用。

**项目地址**：https://github.com/maxma615/yolo26-detect-depth-demo （MIT）

---

## 先看效果

**Web 面板**（浏览器打开 `:8080`，左侧指标 / 中间实时双画面 / 右侧二维码）：

![web 面板](https://raw.githubusercontent.com/maxma615/yolo26-detect-depth-demo/main/docs/images/web_dashboard.png)

**合成输出**：上 = 相机 + 检测框 + person 跟踪（ByteTrack），下 = 深度伪彩 + 距离网格：

![合成输出](https://raw.githubusercontent.com/maxma615/yolo26-detect-depth-demo/main/docs/images/composite_detect_depth.jpg)

## 它做了什么

- **双模型并发**：检测（YOLO26, 640x640 NV12）+ 深度（YOLO26-Depth, 768x768）用 `hobot_dnn` C API
  提交两个异步 UCP 任务，同时驻留单核 BPU 调度器；一个模型的 CPU 前/后处理和另一个模型的 BPU 推理重叠。
- **person 跟踪**：原生 C++ **ByteTrack**（Kalman + 确定性 Hungarian），不依赖 Python。
- **深度距离网格**：把深度图按交点取 3x3 中值输出相对距离（也可 `--depth-meters` 标定成米），
  近=暖色 / 远=冷色。
- **两种显示**：HTTP/MJPEG（浏览器面板，实时刷新指标）+ HDMI 直出（DRM/KMS vsync 翻页，可开机自启）。
- **EMEET PIXY 支持**：任意 `--source` 自动走 V4L2；PIXY 可开**固件级 AI 人形跟踪**（HID，比软件 PID 稳）。
- **真实 BPU 占用**：直接读 `/sys/devices/system/bpu/ratio` 硬件计数，不是估算。

## 实测数据（S100P，BPU@1.5GHz，1080p@30 稳态）

| 指标 | 数值 |
|---|---|
| 相机 / 检测 / 深度帧 | 1:1 零丢帧 |
| 端到端 | ~29.5 fps |
| 检测 BPU | avg ~9.2ms（p95 ~9.4） |
| 深度 BPU | avg ~15.8ms（p95 ~16.4） |
| BPU 占用 | ~65% |
| 显示流 | ~50 fps |

1080p@30 性能基本不变（合成下采样、模型输入固定），**1080p 几乎免费**。

## 快速上手

```bash
git clone https://github.com/maxma615/yolo26-detect-depth-demo.git
cd yolo26-detect-depth-demo

# 1. 模型不进 git，从你已有的板端目录或 URL 拉取
MODEL_SRC=/userdata/yolo26_dual_demo/models ./scripts/download_models.sh

# 2. 部署 + 板端编译
BOARD=root@192.168.x.x ./scripts/deploy.sh

# 3. 运行
bash scripts/run.sh                          # 浏览器 :8080
bash scripts/run.sh --source 2 --cam-fps 60  # 选相机 / 帧率
bash scripts/run.sh --hdmi                   # HDMI 直出
```

浏览器打开 `http://<板ip>:8080/` 即可。

## 打包 .deb（可选）

```bash
cd cpp/build && cpack -G DEB && sudo dpkg -i yolo26-detect-depth-demo_*.deb
```

## 仓库结构

```
cpp/        CMake + main.cpp + kms_display.h + tracking(ByteTrack) + ui + tests
web/        浏览器面板（服务端每次请求重读，改完刷新即生效）
scripts/    run / deploy / download_models / install_boot / pixy_tracking
docs/       architecture / web-ui / hdmi / camera-pixy / packaging
```

## 测试

```bash
cd cpp/build && ctest --output-on-failure   # 3/3 通过
```

---

**欢迎交流**：有改进建议（尤其 BPU 调度 / 深度标定 / 更高帧率）欢迎提 issue 或回帖。
如果觉得有用，给个 star ⭐ 吧～
