# ShotCraft：短视频局部重创作系统

面向短视频后期中的穿帮物体清理和局部元素替换。围绕编辑请求、镜头调度、掩码控制、生成适配、区域保留与可追溯评测设计研究流程。底层生成组件使用 [TencentARC/VideoPainter](https://github.com/TencentARC/VideoPainter)，保留上游来源和许可。

## 当前状态

- 已实现编辑请求校验、跨镜头窗口规划、VideoPainter 直接适配、输入与权重哈希校验、逐步骤资源记录。
- 真实推理已跑通：9 帧、480×720、8 步，推理 477.98 秒。该候选存在车辆残留和重复纹理，不能称为物体移除成功。
- 新增 PyTorch 区域约束、内侧羽化、传统基线与输入拷贝负对照，以及背景/目标区域/边界偏差诊断；33 项测试通过。
- 时序掩码残差网络已实现但尚未训练；多场景效果对照、模型质量改进与应用界面仍未完成。

## 实验与代码

[首次真实推理与内存记录](reports/memory-v1/README.md) · [质量诊断与失败分析](reports/evaluation-v1/README.md) · [8/30 步对照](reports/steps30-v1/README.md) · [系统设计](docs/design.zh-CN.md) · [版本历史](VERSION_HISTORY.md)

输入准备：scripts/prepare_case.py；推理：scripts/run_baseline.py；后台运行与超时保护：scripts/continue_baseline.py；区域约束与指标：src/shotcraft/evaluation.py；离线评测：scripts/evaluate_run.py；VAE 排查：scripts/diagnose_vae.py。源码、配置、种子、输入哈希、原始输出与失败日志均保存。

## 运行环境

Windows，Python 3.11.9，PyTorch 2.4.0，torchvision 0.19.0，transformers 4.42.2，accelerate 0.34.2。生成后端固定于 third_party/VideoPainter 子模块的提交，精确依赖记录见 requirements-lock-windows.txt。

在 VS Code 选择项目解释器后，可运行 python -m unittest discover -s tests -v。生成前需准备清单对应权重及真实输入；每次推理与评测使用新的输出目录。

## 结论边界

当前只有单个开发片段的运行与失败诊断。区域融合可强制保留背景，但不能证明目标被正确移除；已有误差指标不替代语义、感知或时序质量评测。VideoPainter 的模型结构与预训练权重属于上游成果，本项目不声称原创生成主干。上游许可限学术、科研和教育用途，当前按研究原型开展。

30 步对照已完成，车辆残留与重复纹理仍存在，未验证质量改善；详情见 reports/steps30-v1。
