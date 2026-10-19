# 模型部署：从训练权重到稳定推理

部署阶段最常见的 bug 不是模型精度不够，而是**训练和推理的预处理/后处理不一致**。本文按"导出 → 构建 → 校验 → 调优"的顺序整理。

## 1. 导出 ONNX

以 PyTorch 为例：

```python
import torch

model.eval()
dummy = torch.randn(1, 3, 640, 640).cuda()
torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=["images"], output_names=["output"],
    opset_version=17,
    dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
    do_constant_folding=True,
)
```

要点：

- **`model.eval()` 必须调用**，否则 BN/Dropout 行为不同。导出后精度掉一截，先查这个。
- `opset_version` 不是越高越好：TensorRT 对某些新算子支持滞后，17 是较稳的折中（具体看 TRT 版本）。
- `dynamic_axes` 只在真的需要动态 batch/尺寸时加。**静态 shape 更容易优化**，能静态就静态。
- 导出后立刻用 `onnx.checker.check_model` + `onnxruntime` 跑一遍，与 PyTorch 输出对比 `max_abs_diff`，一般应 < 1e-4。

## 2. TensorRT 构建

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("model.onnx", "rb") as f:
    assert parser.parse(f.read()), parser.get_error(0)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

profile = builder.create_optimization_profile()
profile.set_shape("images", (1, 3, 640, 640), (4, 3, 640, 640), (8, 3, 640, 640))
config.add_optimization_profile(profile)

engine = builder.build_serialized_network(network, config)
```

- **FP16**：`config.set_flag(trt.BuilderFlag.FP16)`，多数视觉模型精度损失 < 0.5 个点，速度提升明显。
- **INT8**：需要校准集（`IInt8EntropyCalibrator2`），校准集必须**来自真实场景分布**，否则掉点严重。上线前必须用实测集验证。
- Engine **绑定具体 GPU 型号和 TRT 版本**，换机器必须重建。构建耗时可达数分钟，要缓存。
- `workspace` 太小会导致层回退到慢速实现，构建日志里会有提示。

## 3. 一致性校验（最容易翻车的一步）

训练框架（PyTorch/Ultralytics）通常内置了预处理和后处理，而部署端要自己实现。两边必须逐项对齐：

| 环节 | 常见差异 |
| :--- | :--- |
| 颜色通道 | BGR（OpenCV）vs RGB（PIL/torchvision） |
| 归一化 | `x/255` 后再减均值除方差，顺序和数值不同 |
| Letterbox | 填充值（114 vs 0）、是否居中、缩放取整规则 |
| NMS | IoU 阈值、是否类内 NMS、坐标是归一化还是像素 |
| 输出解析 | 是否转置、是否含 objectness、anchor 顺序 |

**校验方法**：同一张图，分别过 PyTorch 和部署引擎，比较最终框的坐标与分数。允许 1~2 px 偏差，超过就说明某处不一致。

```python
# 一致性检查的最小实现
torch_out = run_torch(img)
trt_out = run_engine(img)
print("max diff:", (torch_out - trt_out).abs().max())
```

## 4. 性能指标

不要只报"平均延迟"，至少给：

- **P50 / P95 / P99 延迟**：平均值会掩盖长尾。
- **吞吐（FPS）**：注意是单卡单流还是多流并发。
- **端到端 vs 纯推理**：预处理（resize/归一化）在 CPU 上经常比推理还慢。
- **显存占用**：多路视频流时这是硬约束。

测量时预热 20~50 次再计时，且要 `torch.cuda.synchronize()` / `cudaStreamSynchronize`，否则测出来的是**入队时间**。

## 5. 优化顺序（按收益排）

1. 先量：定位瓶颈在解码、预处理、推理还是后处理。
2. 预处理搬到 GPU（`cv2.cuda` / DALI / 自写 kernel）。
3. FP16；INT8（需校准与验证）。
4. 降低输入分辨率（往往最有效，且对精度影响可控）。
5. 多流并发 / 批处理，提高 GPU 利用率。
6. 算子融合、去掉不必要的 transpose/contiguous。

## 6. 常见坑

- **engine 与 GPU 不匹配**：在 A100 构建的 engine 拿到 4090 上加载直接失败或结果错误。
- **动态 shape 没设 profile 边界**：运行时报 `invalid shape`。
- **归一化参数写死成 ImageNet 均值**，但训练时用的是自定义均值。
- **NMS 阈值不一致**：训练时 0.45，部署时 0.5，指标对不上还找不到原因。
- **在 CPU 上 benchmark 却声称 GPU 性能**：务必记录硬件型号、驱动、CUDA、TRT 版本。

## 7. 上线前检查清单

- [ ] ONNX 与 PyTorch 输出 `max_abs_diff` < 1e-4
- [ ] 引擎与框架的最终检测框逐一对比通过
- [ ] 在目标硬件上测过 P50/P95/P99 与显存
- [ ] 校准集（INT8 时）与实测集分离
- [ ] 版本矩阵记录：GPU 型号 / 驱动 / CUDA / cuDNN / TRT / 模型哈希
