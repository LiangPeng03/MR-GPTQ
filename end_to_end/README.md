# End-to-End Benchmark & Validation Scripts

两条实验链的入口脚本。环境：conda env `vptq`
（`/home/pengliang/.conda/envs/vptq`，vLLM 为 editable 安装，
源码 `/home/pengliang/vllm-fpquant`）。

## 链 ① vLLM 端到端（真实部署吞吐，论文主表）

| 步骤 | 命令 | 说明 |
|---|---|---|
| 改 CUDA kernel 后重编译 | `bash end_to_end/rebuild_vllm.sh` | GCC13 shim + ninja 增量，只重编 csrc |
| bit-exact 验证（必做） | 见下方 `validate_blite.py` 用法 | 新旧 kernel 输出逐位比对 |
| 算子级 profile | `python end_to_end/bench_blite.py <ours_model_dir>` | 单算子时间 + kernel 分解 |
| 端到端吞吐 | `bash end_to_end/run_blite_bench.sh` | Qwen3-8B ours-lss, BS 1-16, CUDA Graph, 结果落 `benchmark_results/blite_e2e/` |

`validate_blite.py` 两阶段用法（需要旧 `.so` 作参考，`cp` 备份先行）：

```bash
SO=/home/pengliang/vllm-fpquant/vllm/_C.abi3.so
cp $SO /tmp/so_backup.abi3.so
python validate_blite.py phase1        # 旧库生成参考 → /tmp/blite_ref.pt
cp /home/pengliang/vllm-fpquant/build/temp.linux-x86_64-cpython-310/_C.abi3.so $SO
python validate_blite.py phase2        # 新库对比，须打印 ALL_EQUAL: True
```

5 模型完整对比表（fp16/rtn/gptq/mr_gptq/ours）需要 `e2e_models/` 下
的全部模型目录；RTN/GPTQ/MR-GPTQ 基线已于 2026-09-06 从磁盘删除，
如需重测须先用 `model_quant.py` 重新导出。

## 链 ② HF Transformer（模拟量化 + 真实 NVFP4 GEMM，4/6 对比）

单脚本双 arm，结果自动落 `benchmark_results/hf_packed_<arm>_<时间戳>/`：

```bash
# Ours (LSS)：打包权重 + vLLM LSS activation kernel + QuTLASS GEMM
python end_to_end/benchmark_hf_packed.py --arm lss

# True 4/6：权重侧离线 MSE 搜索 + 激活侧运行时 2 候选 MSE 搜索（非融合，
# eager PyTorch——当前不存在任何 4/6 融合部署 kernel）
python end_to_end/benchmark_hf_packed.py --arm four_over_six
```

每个 arm 启动时会打印 `SELF_CHECK rel_err=...`（打包 GEMM vs 已验证
BF16 反量化参考，须 PASS：LSS <2%，4/6 应≈0）。协议：512 in / 128
greedy out / warmup 2 / repeat 5 / BS 1,2,4,8,16；约 20–40 分钟/arm
（4/6 激活侧搜索慢，BS16 每 repeat 约 100 s）。

## 复现结果索引（benchmark_results/）

| 目录 | 内容 |
|---|---|
| `qwen3_8b_final_lss_20260905_014249/` | B-lite 前的 5 模型 vLLM 主表基线 |
| `final_lss_blite_20260905/` | B-lite 后 fp16+ours（论文主表） |
| `hf_packed_lss_20260906_160436/` | HF 链 Ours-LSS |
| `hf_packed_four_over_six_20260906_163857/` | HF 链真实 4/6 |
| `hf_packed_lss_bs1_clean/` | LSS BS1 干净显存重测 |
| `four_over_six_rtn_hf_20260814_173348/` | 8/14 旧 4/6 基线（口径不一致，仅存档） |

## git 同步注意

大文件不要入库（仓库根 `.gitignore` 建议追加）：

```
e2e_models/
benchmark_results/
diagnostics/
*.log
*.pt
```

本目录 5 个脚本均为纯文本，可直接同步。
