# 单窗口特征缓存的内存问题

CFG=6、9 帧、8 步实验完成第一步后，在第二步的 FFN 激活分配处 OOM。after_load 分配约 208 MiB；第一步结束分配约 2.38 GiB；失败时分配约 3.98 GiB，峰值约 4.20 GiB。原日志仍报告一定空闲显存，因此还不能将所有分配失败归结为物理容量。

## 已定位的无用保留

固定上游 pipeline 每次调用 transformer 都设置 return_hidden_states=True 与 return_resample_mask=True。transformer 对所有 42 层执行 torch.cat 并加入 hidden_states_list。该列表只有跨窗口继承时才需要，当前单窗口调用不使用它，但 Python 变量会在下一个 forward 返回之前保留上一轮列表。

在当前 CFG 双路 batch=2、4050 视频 token + 226 文本 token、hidden_dim=3072、BF16 配置下，42 层仅这些特征约占 2,206,826,496 字节（约 2.06 GiB），尚未计入其他激活和权重。它是源码确认的额外保留，实际 OOM 的全部因果仍需完整权重对照验证。

## 修复边界

不修改上游子模块或模型权重。仅在本适配层保证单窗口、prev_clip_weight=0、无 ID 重采样的路径，使用临时 PyTorch hooks 禁止返回逐层特征和重采样掩码，再补回调用方期望的三元组结构。跨窗口/ID 路径明确拒绝使用该优化；正常或异常退出都移除 hooks。生成服务若并发共享同一模型，仍需串行化，本优化不提供线程安全保证。

35 项测试通过。真实 CogVideoX 小结构（2 层）比较修复前后噪声预测逐元素完全相同，且退出后原返回行为恢复；这不是完整 5B 权重数值对照。

来源：[固定 pipeline](https://github.com/TencentARC/VideoPainter/blob/bbab6cd5cd5cb89f0e2444305c32fd74a010ae0a/diffusers/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_inpainting_i2v_branch_anyl.py)、[固定 transformer](https://github.com/TencentARC/VideoPainter/blob/bbab6cd5cd5cb89f0e2444305c32fd74a010ae0a/diffusers/src/diffusers/models/transformers/cogvideox_transformer_3d.py)。

## 完整权重验证

使用相同输入、seed=42、guidance=6、8 步、内存高效注意力，新输出目录 D:/career-projects/runs/ShotCraft/20261029-guidance6-no-cache-v1。结果未完成前不声称修复 OOM 或改善画质。原有失败文件与日志保留。
