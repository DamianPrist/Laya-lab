# jev-decision-lab

用于比较本地 Laya 与 DeepSeek 在同一批选择题上的判断准确率和响应延迟，并开展 Laya 选项位置相关实验。

## 保留的策略

- `raw-jev`：题目和候选项直接交给 Decision Model。当前 `DECISION_BACKEND=laya` 时调用本地 Laya。
- `llm-only`：题目和候选项直接交给 DeepSeek。
- `compare`：对同一输入分别运行以上两条独立路线，便于配对比较；不代表一个 Laya + DeepSeek 的串联推理策略。

未进行实验的 Evidence + Jev 混合路线已从项目中移除。

## 环境配置

需要 Python 3.11+。创建环境并安装依赖：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中配置所需服务。`raw-jev` 使用 Laya 时不需要 DeepSeek 配置；`llm-only` 需要 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_MODEL`；`compare` 两者都需要。不要提交 `.env`。

## Laya 服务

Laya 通过本地 HTTP 服务运行。英文实验可预加载 English checkpoint：

```powershell
$env:LAYA_DEVICE="cuda"
$env:LAYA_PRELOAD="1"
$env:LAYA_MODELS="english"
laya-serve
```

在另一终端检查服务：

```powershell
python main.py --check
```

## 运行

```powershell
# 单独运行 Laya
python main.py --batch cases.json --strategy raw-jev

# 单独运行 DeepSeek
python main.py --batch cases.json --strategy llm-only

# Phase 1：同一题目分别运行 Laya 与 DeepSeek
python main.py --batch cases.json --strategy compare
```

不指定策略时默认使用 `compare`。普通 CLI 结果会追加到 `results.jsonl`；正式实验使用独立输出文件，避免与普通运行记录混合。

## 已完成实验数据

- `phase1_laya_vs_deepseek_20260928T161819.jsonl`：Phase 1 正式对比。
- `position_bias_laya_20260928T165731.jsonl`：选项位置偏向实验。
- `laya_label_vs_sequence_20260928T172328.jsonl` 和对应 `.summary.json`：A 标签与序列位置区分实验。
- `smoke_cases.json`、`smoke_results.jsonl`：Smoke Test 输入和记录。

正式实验题库为 `cases.json`。位置实验脚本为 `run_position_bias_experiment.py` 和 `run_label_vs_sequence_experiment.py`。

## 项目结构

```text
jev-decision-lab/
├── main.py
├── config.py
├── models.py
├── experiment.py
├── clients/
│   ├── jev_client.py
│   └── deepseek_client.py
├── strategies/
│   ├── base.py
│   ├── raw_jev.py
│   └── llm_only.py
├── tests/
├── cases.json
├── run_position_bias_experiment.py
└── run_label_vs_sequence_experiment.py
```
