# Controller QLoRA/SFT 五小时 Pilot

> 目标：在单张 A100 80GB 上训练一个 Controller-only adapter，并在相同、已审核的
> Controller states 上比较 base 与 SFT policy。本文不训练 Planner、Reader、Checker、
> Answerer 或 DPO，也不读取 Test 个案。

## 1. 冻结边界

- 基座：`/workspace/models/Qwen3.5-27B`。
- 数据：只接受 `component=controller` 且经过 Teacher review/export 的
  `controller_sft.jsonl`。
- 训练消息与线上消息完全相同：system 为冻结的 Controller Prompt，user 为
  `build_controller_user_prompt()` 产生的缩进 JSON，assistant 为一个合法动作 JSON。
- 第一轮只做 60 optimizer steps。该配置用于验证闭环和方向，不作为最终超参数结论。
- `Test` 不参与数据制作、checkpoint 选择或离线 policy 对比。

## 2. 训练前硬门槛

```bash
softdoc teacher-data audit-controller \
  /workspace/data/controller-pilot/controller_sft.jsonl \
  --output /workspace/data/controller-pilot/controller_sft_audit.json

python scripts/train_sft.py \
  --data /workspace/data/controller-pilot/controller_sft.jsonl \
  --require-component controller \
  --validate-only
```

两条命令都必须成功。审计报告要求 ControllerInput、Action schema、可见 ID、唯一
example ID 和 state-action 去重全部通过。正确最终答案不等于高质量轨迹；幸运猜对、
错误 Evidence、无意义循环和上游 Reader/Checker 污染后的状态不能作为正样本。

## 3. 限时 QLoRA 命令

训练前停止占用同一张 GPU 的 vLLM 服务，但不要删除模型或缓存：

```bash
python scripts/train_sft.py \
  --data /workspace/data/controller-pilot/controller_sft.jsonl \
  --require-component controller \
  --model /workspace/models/Qwen3.5-27B \
  --model-class image_text_to_text \
  --output /workspace/runs/controller-qlora-pilot-v0-1 \
  --qlora --bf16 --gradient-checkpointing \
  --max-length 8192 \
  --max-steps 60 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --lr-scheduler-type cosine \
  --optim paged_adamw_8bit \
  --save-steps 20 \
  --save-total-limit 3 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,out_proj
```

配置镜像保存在
`configs/training/controller_qlora_pilot_v0_1.json`。以 100--180 条、长度约
2.5k--6k token 的决策样本估计，模型加载加 60 步训练通常应控制在约 45--120 分钟；
这是工程估计，首次实际运行必须记录峰值显存、tokens/s 和墙钟时间。

若 4-bit 模型类或 bitsandbytes 初始化失败，先确认服务器的 Transformers/PEFT/
bitsandbytes 与环境报告一致，不要直接切换为全参数训练。时间不足时的降级顺序是：

1. 保持 27B、rank 16 和 8192 context，把 `max_steps` 降为 40；
2. 若显存不足，把 `max_length` 降为 6144，并先统计被截断样本；
3. 仍无法启动才改用已缓存的 Qwen 8B 做 pipeline smoke；8B 结果不能冒充 27B 实验。

## 4. 只让 Controller 加载 adapter

vLLM 必须同时暴露 base 模型名和一个 LoRA alias。示意命令：

```bash
vllm serve /workspace/models/Qwen3.5-27B \
  --served-model-name /workspace/models/Qwen3.5-27B \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank 16 \
  --lora-modules controller-sft=/workspace/runs/controller-qlora-pilot-v0-1/adapter
```

运行器新增了向后兼容的 `--controller-model`：

```bash
python scripts/run_model_batch.py \
  ...原有冻结参数... \
  --text-model /workspace/models/Qwen3.5-27B \
  --controller-model controller-sft
```

只有 Controller 请求 `controller-sft`；其余四个 Agent 仍请求 base。省略
`--controller-model` 时自动回退到 `--text-model`，旧 baseline 行为不变。batch
manifest 会同时记录二者，防止错误地把“全 Agent adapter”当成 Controller-only。

## 5. 离线前后对比

同一个 vLLM 服务完成 base/adapter 对照，不执行真实检索，也不读取 Gold answer：

```bash
python scripts/evaluate_controller_sft_offline.py \
  --data /workspace/data/controller-pilot/controller_sft.jsonl \
  --model base=/workspace/models/Qwen3.5-27B \
  --model sft=controller-sft \
  --base-url http://127.0.0.1:8000/v1 \
  --output /workspace/runs/controller-policy-offline-v0-1
```

`summary.json` 报告：首轮 JSON 合法率、Action schema 合法率、可见 ID 合法率、
repair 后最终合法率、Teacher exact/route agreement、连续结构动作重复率、动作分布和
平均延迟。`route agreement` 忽略自由文本 `local_problem` 和 `SEARCH new` query 的
措辞，但不忽略动作类型、operation 或 typed handle。

离线 agreement 只能证明 Student 学会 Teacher policy，不能证明闭环 QA 提升；最终仍需
在冻结集合上以真实 Environment 做 paired evaluation。
