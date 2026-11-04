# verl-tinker 本地服务接口手册

**核对日期**　2026-11-04。面向已启动的服务，覆盖 22 个业务路径、请求字段、有效默认值、返回值、Python SDK 和原始 HTTP 的用法。服务参考版本为 `verl-project/verl-recipe@0a9360bb366d8bd71e26fb86da1acf077da9e6d9`。该提交服务端锁定 `tinker==0.19.0`，客户端示例锁定 `tinker==0.24.0`、`tinker-cookbook==0.5.2`。文中“本版”均指这组版本。未连接你的内网实例。下面的只读命令能确认本机差异。

## 1　先从运行实例拿到真实接口

在与服务同机、端口可达的 shell 执行。若在 Z-Desktop 容器或另一台机器运行客户端，换成从该进程可达的服务地址。SDK 的根地址以站点根路径结尾，SDK 会自行追加 `/api/v1/...`。

~~~bash
export TINKER_BASE_URL=http://127.0.0.1:8000/
export TINKER_API_KEY=tml-verl-tinker-local
ROOT="${TINKER_BASE_URL%/}"

curl -fsS "$ROOT/api/v1/healthz" | jq
curl -fsS "$ROOT/api/v1/get_server_capabilities" | jq
curl -fsS "$ROOT/openapi.json" -o verl-tinker-openapi.json
jq -r '.paths | to_entries[] | select(.key | startswith("/api/v1/")) |
       .key as $p | .value | keys[] | "\(. | ascii_upcase) \($p)"' verl-tinker-openapi.json
jq '.paths["/api/v1/asample"].post.requestBody, .components.schemas.SampleRequest' verl-tinker-openapi.json

# 在你部署时用的源码仓库内运行
git rev-parse HEAD
# 在相应的服务端、客户端 Python 环境分别运行
python -c 'from importlib.metadata import version; print(version("tinker"))'
~~~

浏览器可打开 `http://127.0.0.1:8000/docs`。若用了代理，需带实际挂载前缀。OpenAPI 能显示类型，但若路由收的是通用 `dict`，页面不会展开内部字段。量化训练模式、Teacher 配置或 LoRA rank 也不能通过 Swagger 判断。`healthz` 应返回 `{"status":"ready"}`。`supported_models[].model_name` 给出可传的名称。该实现用 rollout 的 `max_response_length` 填写 actor 的 `max_context_length`，取不到时回退为 4096，读到这个值也要和实际训练、推理配置对照。兼容 key 满足 SDK 需要，网关没有据此提供用户鉴权。

## 2　HTTP 完整目录

以下路径都在 `/api/v1/` 下。除标注为 GET 外均为 POST。无 `?` 的参数为必要输入，`?` 表示可省；未另注的可空字段默认为 `null`。路径参数直接写在 URL 中，POST 的结构化输入放在 JSON 请求体中。下表写的是绑定到 GPU 实现的行为。固定 `type` 标签如果使用，可省略并采用请求类型默认值，详见本节末尾。

### 2.1　服务、会话及模型

| 路径与方法 | 输入 | 结果与用法 |
|---|---|---|
| `GET healthz` | 无 | `status` 可能是 `initializing`、`ready`、`error`、`shutdown_complete`；初始化异常有 `message`。 |
| `GET/POST get_server_capabilities` | 无 | `supported_models`，每项含 `model_name`、`max_context_length`。actor 在前；已配置 Teacher 可能按名称和权重路径各出现一次。不能据此判定有哪些 worker。 |
| `POST client/config` | 无 | SDK 连接配置。此版本 `pjwt_auth_enabled=false`、`credential_default_source="api_key"`、`sample_dispatch_bytes_semaphore_size=0`、`inflight_response_bytes_semaphore_size=0`、`parallel_fwdbwd_chunks=false`。 |
| `POST create_session` | 无或 `{}`；也接收 SDK 的 `tags`、`user_metadata`、`sdk_version`、`project_id`、`type` | 直接返回 `session_id`、`type` 及 `info_message/warning_message/error_message`（本版为 null）。额外字段不参与资源隔离。 |
| `POST sessions` | 无 | `{"sessions":[...]}`。本版返回全部 ID。示例 SDK 的 `RestClient.list_sessions` 使用 GET，和该路由方法不同；直接 POST。 |
| `GET sessions/{session_id}` | 路径 `session_id:str` | `training_run_ids`、`sampler_ids`，后者仅含仍有效的采样资源。未知会话返回 404。 |
| `POST session_heartbeat` | 空或对象；SDK 发 `session_id`、`type` | 直接返回 `{"type":"session_heartbeat"}`。 |
| `POST telemetry` | 空或对象；SDK 常发 `events`、`platform`、`sdk_version`、`session_id`、`process_uuid` | 仅确认 `{"status":"accepted"}`。不能通过该接口查询遥测数据。 |
| `POST create_model` | `session_id:str`、`model_seq_id:int`、`base_model:str`；`lora_config?:object`、`user_metadata?:object` | 返回 `request_id`、`model_id`；轮询结果也含 `model_id`。同一 session 中使用逻辑序号。只建立已加载 actor 的训练句柄。 |
| `POST get_info` | `model_id:str` | 直接返回 `model_id`、`model_name`、`model_data.{arch,model_name,tokenizer_id}` 等。此实现固定返回 `is_lora=false`、`lora_rank=null`，不要用它判定实际训练模式。 |
| `POST create_sampling_session` | `session_id:str`、`sampling_session_seq_id:int`；`base_model?:str`、`model_path?:str` | 至少给一个有效目标。返回 `sampling_session_id`。基础模型或 Teacher 常用 `base_model`；已同步快照用 `model_path`，传回服务给出的完整 URI。 |
| `GET samplers/{sampler_id}` | 路径 `sampler_id:str` | 返回 `sampler_id`、`base_model`、`model_path`；未知句柄返回 404。记录存在不保证旧权重仍驻留。 |
| `POST shutdown` | 无 | `{"status":"accepted"}`，触发整个服务关闭。日常客户端退出无需调用。 |

`lora_config` 的对象字段为 `rank:int`（对象内必填）、`seed:int|null=null`、`train_mlp:bool=true`、`train_attn:bool=true`、`train_unembed:bool=true`。Python 工厂方法默认 `rank=32`。参数能通过协议，但不会热配置训练引擎；全参或 LoRA、rank、目标模块和适配器路径以启动 YAML 的 `actor_rollout_ref.model` 为准。若要区分一次实验和另一次实验，记住不同 `model_id` 共用物理 actor 和优化器。多个用户的训练操作要由上层排队，或部署独立实例。

### 2.2　训练、采样、保存和结果领取

| 路径 | 必填、可选参数 | 成功结果 |
|---|---|---|
| `forward` | `model_id:str`、`forward_input:{data,loss_fn,loss_fn_config?}`、`seq_id?:int` | 异步。只前向，返回每个样本的 `loss_fn_outputs[i].logprobs`，`loss_fn_output_type="log_probs"`，`metrics={}`。请求结构要求写 `loss_fn`，执行只使用 `data`。 |
| `forward_backward` | `model_id:str`、`forward_backward_input:{data,loss_fn,loss_fn_config?}`、`seq_id?:int` | 异步。计算并累积梯度。输出 `loss_fn_outputs`、`loss_fn_output_type`、`metrics`；总损失看 `metrics["loss:sum"]`。有些后端路径拿不到真实反传 logprobs 时会回填旧值或 0，要取准确当前值用 `forward`。 |
| `optim_step` | `model_id:str`、`adam_params:object`、`seq_id?:int` | 异步。更新 actor，返回 `type="optim_step"` 和 `metrics`。本版 HTTP 字段名为 `adam_params`。 |
| `asample` | `prompt:ModelInput`、`sampling_params:SamplingParams`；`num_samples?:int=1`、`sampling_session_id?:str`、`base_model?:str`、`model_path?:str`、`seq_id?:int`、`prompt_logprobs?:bool=null`、`topk_prompt_logprobs?:int=0` | 异步。指定已有采样 ID，或直接指定模型目标。返回 `type="sample"`、`sequences`、`prompt_logprobs`、`topk_prompt_logprobs`。详见第 4 节。 |
| `save_weights` | `model_id:str`（先创建模型）；`path?:str` 为名称，缺省自动生成。SDK 另可发 `seq_id`、`ttl_seconds`，0.24 客户端还可发 `overwrite=false` | 异步。将训练 checkpoint 写入服务器 `checkpoint_dir`，客户端收到 `tinker://verl-tinker/state/<名称>`。本版处理器不执行 TTL，也不按 overwrite 提供同名保护，请用唯一名称。 |
| `load_weights` | `path:str` 为训练态 URI；`optimizer?:bool=true` 是原始 HTTP 缺省值。SDK 同时传 `model_id`、`seq_id`，可传 `weights_access_token` | 异步。恢复 actor，返回 `path`。本版 worker 始终调用 `load_checkpoint`；`optimizer=false` 分支额外清梯度，不能据此保证丢弃 Adam 一、二阶历史。同版本状态还可能跳过重复加载。 |
| `weights_info` | `tinker_path:str` 或别名 `path:str` | 直接返回 checkpoint 的 `base_model` 等元数据。本版给 Cookbook 的兼容值为 `is_lora=true`、`lora_rank=1`、三个 `train_* = true`，并不反映实际训练配置。 |
| `save_weights_for_sampler` | `model_id:str`；`path?:str`、`sampling_session_seq_id?:int`、`seq_id?:int`、`ttl_seconds?:int`。省略 `path` 时必须有 `sampling_session_seq_id` | 异步。同步 actor → rollout。传 `path` 走具名分支，返回自动生成的 `tinker://verl-tinker/weights/step-…`；服务端不沿用传入的名字。不传 `path` 的分支返回 `sampling_session_id`。两者都受单 rollout 版本限制。 |
| `retrieve_future` | `request_id:str`；`allow_metadata_only?:bool=false` 本版忽略 | 直接回最终业务对象、`{"type":"try_again",...}` 或 `{"error":...,"category":"Application"}`。单次可等待约 30 秒。结果仅可成功领取一次。 |

采用 SDK 固定标签的请求，`get_info/create_model/create_sampling_session/optim_step/save_weights_for_sampler/save_weights/load_weights` 的 `type` 默认为对应路径名；`asample` 的 `type` 默认为 `"sample"`。`forward` 和 `forward_backward` 的本地数据类无须传 `type`。`seq_id` 是请求序号；本版多个训练操作直接生成随机 `request_id`，切勿按相同 `seq_id` 自动重发更新。`create_session`、心跳和遥测直接接受字典，无必填业务字段。`save_weights`、`load_weights`、`weights_info` 的表格列出了处理器实际读取的字段及常见兼容字段。

README 还列有 `POST /api/v1/export_model`，本次提交的实际路由没有注册该路径。训练态 URI 可按配套方法恢复，不能当作 Hugging Face `save_pretrained` 目录。源码 `schemas.py` 中出现的旧 `/v1/compute_values` 等名字也不能当作已对外开放的路由。

## 3　训练数据、损失和优化器

### 一个 Datum 的结构

`data` 为非空样本列表。每个元素包含 `model_input` 和 `loss_fn_inputs`。文本输入先用模型匹配的 tokenizer 转成 token ID，`ModelInput.from_ints(ids)` 生成 `{"chunks":[{"type":"encoded_text","tokens":[...]}]}`。网关调用 `to_ints()`，只支持这种 token 块。下面是可以直接放进 HTTP 的一个三目标位置的 SFT 样本，数字仅示意结构。

~~~json
{
  "model_input": {"chunks": [{"type": "encoded_text", "tokens": [10, 11, 12]}]},
  "loss_fn_inputs": {
    "target_tokens": {"data": [11, 12, 13], "dtype": "int64", "shape": [3]},
    "weights": {"data": [0.0, 1.0, 1.0], "dtype": "float32", "shape": [3]}
  }
}
~~~

序列 `x` 有 `N+1` 个 token 时，`model_input=x[:-1]`、`target_tokens=x[1:]` 均长 `N`。示例表示不给预测 11 计损失，给预测 12、13 计损失。SDK 允许给标准键直接传 Python 列表。HTTP 张量推荐用 `data` 扁平数组、`dtype`（`int64` 或 `float32`）、`shape`。一维可推断形状；二维须给 `[N,K]`。上游类型也声明了可选 `sparse_crow_indices`、`sparse_col_indices`，本版本地转换器只把 `data` 按 `shape` 重排，二维 Top-K 请发稠密 `TensorData`。
 
| `loss_fn` | `loss_fn_inputs` | `loss_fn_config` 与实际含义 |
|---|---|---|
| `cross_entropy`，普通 SFT | `target_tokens:int64[N]`、`weights:float32[N]`（也读 `mask`） | 留空。按 token 加权求和。对话 SFT 通常仅回答位置权重为 1。 |
| `cross_entropy`，Top-K 蒸馏 | `target_tokens:int64[N,K]`、`weights:float32[N,K]`，同形 | 留空。每个位置给 K 个 Teacher token ID 和对应概率权重；有效行通常在 K 内归一化，整行 0 可屏蔽。发送概率前先处理 Teacher 的对数概率。同一 batch 保持维数一致。 |
| `importance_sampling` | RL 三件套，见下 | 留空。本版映射为 verl 的 token 级 IS / REINFORCE 路径，内部阈值为 2.0。 |
| `ppo` | RL 三件套 | `clip_low_threshold=0.8`、`clip_high_threshold=1.2`。传的是概率比上下界；例如 0.8 与 1.2。 |
| `cispo` | RL 三件套 | 同名的两个阈值，默认 `0.0` 和 `4.0`。 |
| `dro` | RL 三件套 | 必须指定有限正数 `beta`，如 `{"beta":0.05}`。 |
| `custom_from_config` | 走 RL 数据转换，仍需 RL 三件套；具体 loss 能否运行取决于启动配置及所需张量 | verl-tinker 扩展。忽略请求的 `loss_fn_config`，使用 `actor_rollout_ref.actor` 配置。另一个 SDK 函数 `forward_backward_custom` 在客户端用 Python/PyTorch 计算自定义损失，仅本地 1-D 目标路径可用。 |

RL 三件套为 `target_tokens:int64[N]`、采样时行为策略的 `logprobs:float32[N]`、客户端算好的 `advantages:float32[N]`。可选 `response_mask`（首选）、`mask`、`weights` 标出需学习的回答 token；均缺时本版用非零 `advantages` 或 `logprobs` 推断。无有效回答位置会失败。`rollout_is_weights:float32[N]` 是可选的额外重要性权重，同一批应全部给或全部不传。SDK 的列表自动转换只认识部分键，给 `response_mask` 这类扩展键请显式构造 `types.TensorData(data=..., dtype="float32", shape=[N])`。奖励、分组和优势估计由客户端或 Cookbook 完成，不能只给一列 `reward` 把标题写成 GRPO。

对 `ppo` 和 `cispo`，阈值需满足 `0 ≤ low ≤ 1 ≤ high`；未声明的参数会被拒绝。原生 RL loss 的配置关闭了服务端额外 KL 损失及熵项，需要 KL 的 Cookbook 循环可取 reference logprobs 后纳入优势。`forward` 仅支持一维目标，只返回与实际序列对齐的 logprobs。网关用 `model_input` 加 `target_tokens` 的最后一个 token 重建序列，前面的目标应按 next-token 规则对齐。

| `adam_params` 字段 | 类型与默认值 | 用法 |
|---|---|---|
| `learning_rate` | float，`1e-4` | 该次优化步的学习率。 |
| `beta1` | float，`0.9` | 一阶矩衰减。 |
| `beta2` | float，`0.95` | 二阶矩衰减。 |
| `eps` | float，`1e-12` | 数值稳定项；若想用 `1e-8` 请显式传。 |
| `weight_decay` | float，`0.0` | 解耦权重衰减。 |
| `grad_clip_norm` | float，`0.0` | SDK 接受，网关转换时不传给 verl；按服务端配置控制梯度裁剪。 |

`optim_step` 的 `adam_params` 对象本身必填。HTTP 即使发 `{}` 也会套用上述 SDK 类型默认值。若希望保持某个学习率，每步显式传值。训练 batch size 由 `data` 条数及 SDK 分块决定；GPU 数、TP、后端 microbatch 和显存策略由启动配置决定。

## 4　采样、逐 token 概率和 Teacher

`SamplingClient.sample(prompt, num_samples, sampling_params, include_prompt_logprobs=False, topk_prompt_logprobs=0)` 的 Python 签名要求显式给前三项。HTTP 层允许 `num_samples` 缺省为 1。`prompt` 是 token 输入，聊天场景需在客户端先套模型的 chat template。服务不直接接收 `messages`、字符串 prompt 或 `stream=true`，`sample` 等待后一次返回所有序列。

| 字段 | 默认值 | 实际效果 |
|---|---|---|
| `sampling_params.max_tokens` | SDK 为 `None`，网关使用 `256` | 限制新生成 token 数。网关对 `0` 也回退为 256。仅查 prompt 分数用 `compute_logprobs` 或下述 1-token 组合。 |
| `sampling_params.temperature` | `1.0` | 采样温度。常用 0 做贪心、0.7 做随机采样。 |
| `sampling_params.top_p` | `1.0` | 核采样累计概率阈值。 |
| `sampling_params.top_k` | `-1` | 每步采样保留的候选个数，-1 不限。与返回多少个 Teacher 概率的参数不同。 |
| `sampling_params.seed` | `None` | 随机种子。一次请求多条时，第 i 条使用 `seed+i`，i 从 0 开始。 |
| `sampling_params.stop` | `None` | 可传一个字符串、纯字符串列表或纯 token ID 列表。字符串走 `stop`，整数列表走 `stop_token_ids`。本版空列表等同没有额外 stop 规则，字符串停止标记会请求包含在输出中。 |
| `num_samples` | HTTP `1` | 正整数。一次为相同 prompt 发起多条独立生成；同组 RL 样本可用。 |
| `include_prompt_logprobs`（SDK）／`prompt_logprobs`（HTTP） | SDK `False` | 返回输入序列中实际 token 的条件对数概率。首位置没有前文，通常为 `None`。 |
| `topk_prompt_logprobs` | `0` | 大于 0 时，按 prompt 位置返回 `(token_id, logprob)` 列表，首行为 `[]`。该实现走 Top-K 输出分支，不能假定标量 `prompt_logprobs` 同时返回。K 受推理引擎 `max_logprobs` 限制。 |

输出 `sequences[i].tokens` 是新生成的 token ID，`logprobs` 是这些 token 的采样对数概率，`stop_reason` 为 `stop` 或 `length`。输入实际 token 的分数在顶层 `prompt_logprobs`，输入各位置的候选分布在 `topk_prompt_logprobs`。这版没有生成段 `topk_sample_logprobs` 或任意 token ID 打分的 `target_prompt_logprobs` 参数。多样本的 prompt 结果只从第一条响应抽取。

| 任务 | 资源与调用 |
|---|---|
| 对当前训练后策略生成 | 需要 rollout。先 `train.save_weights_and_get_sampling_client()` 同步权重，再 `sample`。 |
| 对基础模型或 Teacher 生成 | `svc.create_sampling_client(base_model=名字)`。基础模型生成要求 rollout 仍保存初始权重；独立 Teacher 有自身推理资源。 |
| 对实际 prompt 逐 token 打分 | `sampler.compute_logprobs(prompt).result()`。底层组合为 `num_samples=1`、`max_tokens=1`、`prompt_logprobs=true`、`topk_prompt_logprobs=0`。匹配的 actor/ref 可以兜底；此时返回的生成段为空。 |
| 取 Teacher Top-K | 对已部署的 Teacher 调 `sample(prompt, 1, SamplingParams(max_tokens=1), topk_prompt_logprobs=K)`。匹配 rollout 也可给 Top-K。纯 actor/ref 的标量兜底无法提供这个结果。 |
| 对旧模型采样 | 一个服务只有一份 rollout 权重。优化后尚未同步时，旧版本可能还在；同步更新替换后，旧句柄可能失效。`base_model=actor_name` 锚定初始版本，不能用来表达“最新训练权重”。 |

## 5　Python SDK 实例

### 5.1　只做采样，不改权重

用仓库的客户端环境：在 `verl_tinker/client_examples` 下用 `uv sync --frozen` 准备锁定依赖，然后用 `uv run python <脚本>` 执行。此例直接从客户端可读的本地模型目录加载 tokenizer，适用于 ModelScope 下载并已复制或挂载的模型。脚本从服务返回的列表取首项作为 actor；若要采 Teacher，显式选它的公开名并使用与它一致的 tokenizer。

~~~bash
export TINKER_BASE_URL=http://127.0.0.1:8000/
export TINKER_API_KEY=tml-verl-tinker-local
export TOKENIZER_DIR=/实际路径/模型目录
~~~

~~~python
import os
import tinker
from tinker import types
from transformers import AutoTokenizer

svc = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
models = svc.get_server_capabilities().supported_models
model = models[0].model_name
print("服务模型", [m.model_name for m in models])
tok = AutoTokenizer.from_pretrained(os.environ["TOKENIZER_DIR"], local_files_only=True)
sampler = svc.create_sampling_client(base_model=model)

messages = [{"role": "user", "content": "用一句话解释变电站的作用。"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt = types.ModelInput.from_ints(tok.encode(text, add_special_tokens=False))
res = sampler.sample(
    prompt=prompt, num_samples=1,
    sampling_params=types.SamplingParams(max_tokens=128, temperature=0.7, top_p=0.9),
).result()
seq = res.sequences[0]
print(tok.decode(seq.tokens, skip_special_tokens=True))
print("停止原因", seq.stop_reason, "生成 token 数", len(seq.tokens))
~~~

示例要求有可用的初始 rollout，且 tokenizer 含对话模板。若部署的是 Base 模型且没有模板，直接 `tok.encode("变电站的主要作用是", add_special_tokens=False)` 构造续写 prompt。若你需要查询的是已有问答中**回答部分**的 logprobs，把 prompt 与该回答一起编码送入 `compute_logprobs`，再按回答起点切片；只发送问题不会返回指定答案的概率。

### 5.2　从零做一步 SFT、存盘，再用新权重采样

这段会实际更新共享 actor 并写一次 checkpoint。即使只用 actor 配置，也可以执行到打印 checkpoint 为止。示例选择文本续写，便于精确控制 prompt/answer 边界；正式对话 SFT 可换用对应 Cookbook renderer。

~~~python
import os
from uuid import uuid4

import tinker
from tinker import types
from transformers import AutoTokenizer

svc = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
model = svc.get_server_capabilities().supported_models[0].model_name
tok = AutoTokenizer.from_pretrained(os.environ["TOKENIZER_DIR"], local_files_only=True)
train = svc.create_lora_training_client(base_model=model)  # 训练模式由服务端决定

def make_datum(question, answer):
    prefix = tok.encode(question, add_special_tokens=False)
    completion = tok.encode(answer, add_special_tokens=False)
    if tok.eos_token_id is not None:
        completion.append(tok.eos_token_id)
    assert prefix and completion
    tokens = prefix + completion
    mask = [0.0] * len(prefix) + [1.0] * len(completion)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": mask[1:]},
    )

question = "问题：变压器的主要作用是什么？\n回答："
batch = [make_datum(question, "在交流系统中变换电压等级。")]
fb = train.forward_backward(batch, loss_fn="cross_entropy").result()
print("总损失", fb.metrics["loss:sum"])
step = train.optim_step(types.AdamParams(learning_rate=1e-5, eps=1e-8)).result()
print("更新指标", step.metrics)
state = train.save_state(name="sft-" + uuid4().hex[:12]).result()
print("训练态 checkpoint", state.path)

# 可选。在 shell 设置 ENABLE_ROLLOUT=1 后才调用，需要 actor + rollout。
if os.getenv("ENABLE_ROLLOUT") == "1":
    latest = train.save_weights_and_get_sampling_client()
    prompt = types.ModelInput.from_ints(tok.encode(question, add_special_tokens=False))
    seq = latest.sample(prompt, 1, types.SamplingParams(max_tokens=96, temperature=0)).result().sequences[0]
    print("更新后回答", tok.decode(seq.tokens, skip_special_tokens=True))
~~~

`forward_backward` 和 `optim_step` 分开，便于累积梯度或保护训练顺序。上例先等前一步成功再更新。若接入异步 FastAPI，`fb = await train.forward_backward_async(batch, "cross_entropy")` 得到 future，接着 `out = await fb.result_async()` 才是训练结果。`await sampler.sample_async(...)` 则直接得到采样结果。

### 5.3　保存、恢复和 SDK 方法对照

| Python 0.24 方法与参数 | 对应操作或返回值 |
|---|---|
| `ServiceClient(user_metadata=None, project_id=None, **kwargs)` | 常用 `base_url`、`api_key`，也可通过环境设置。`get_server_capabilities()` 返回模型列表。 |
| `svc.create_lora_training_client(base_model, rank=32, seed=None, train_mlp=True, train_attn=True, train_unembed=True, user_metadata=None)` | 返回训练句柄。服务端已经加载模型，模型路径和 LoRA 配置不会按每次调用重建。 |
| `svc.create_sampling_client(model_path=None, base_model=None, retry_config=None)` | 返回采样句柄。`retry_config` 调客户端重试，建议使用关键字指定目标。 |
| `train.get_info()`、`train.get_tokenizer()` | 返回模型元信息或 tokenizer。客户端调用后者可能根据公开模型名查找 tokenizer；离线用明确的本地目录。 |
| `train.forward(data, loss_fn, loss_fn_config=None)`、`train.forward_backward(data, loss_fn, loss_fn_config=None)` | 均返回 future，调用 `.result()`。只前向和反传的返回差异见第 2 节。 |
| `train.forward_backward_custom(data, loss_fn, *, loss_type_input="logprobs")` | `loss_fn` 为 Python 回调，取 `(data, logprobs)`，返回可微标量损失和 metrics 字典。与服务端 `custom_from_config` 分开使用。 |
| `train.optim_step(adam_params)` | 返回 future，见第 3 节的 6 个参数。 |
| `train.save_state(name, ttl_seconds=None, overwrite=False)` | 获取持久训练状态 `.result().path`，保留它以恢复。 |
| `train.load_state(path, weights_access_token=None)`、`train.load_state_with_optimizer(path, weights_access_token=None)` | 在现有训练句柄上恢复，均需 `.result()`。二者发送的 `optimizer` 不同，当前后端差异见 `load_weights`。 |
| `svc.create_training_client_from_state(path, user_metadata=None, weights_access_token=None)` 及 `create_training_client_from_state_with_optimizer(...)` | 从已有训练 URI 新建逻辑句柄并加载；同样会改变这个服务的共享 actor。 |
| `train.save_weights_for_sampler(name, ttl_seconds=None)` | 返回 future，取 `.result().path` 后 `svc.create_sampling_client(model_path=path)` 或 `train.create_sampling_client(model_path=path, retry_config=None)`。URI 的可用性受 rollout 版本约束。 |
| `train.save_weights_and_get_sampling_client(name=None, retry_config=None)` | 同步并直接返回采样句柄。0.24 的 `name` 已弃用且不生效，直接无参调用。 |
| `sampler.sample(prompt, num_samples, sampling_params, include_prompt_logprobs=False, topk_prompt_logprobs=0)`、`sampler.compute_logprobs(prompt)` | 同步形式都返回 future；分别是生成和逐 token 标量打分。`get_tokenizer()`、`get_base_model()` 可查 tokenizer 和名称。 |
| `svc.create_rest_client()` | 此本地实现可查 `get_session(session_id)`、`get_sampler(sampler_id)`、`get_weights_info_by_tinker_path(tinker_path)`。云端 checkpoint 发布、下载、计费等方法没有对应本地业务路由。 |

### 5.4　原始 HTTP：不依赖 Tinker SDK 的采样请求

以下使用 Python `requests`，在已就绪且基础权重仍可用于生成的 rollout 上运行。示例 ID 仅用于验证协议，如果需要有意义的答案请替换为前面的 tokenizer 编码。这里直接用 `base_model` 指定对象，无须先建 sampling session。

~~~python
import os
import time
import requests

root = os.environ["TINKER_BASE_URL"].rstrip("/") + "/api/v1"
http = requests.Session()
http.headers["X-API-Key"] = os.environ["TINKER_API_KEY"]

def post(route, body):
    resp = http.post(f"{root}/{route}", json=body, timeout=(5, 45))
    resp.raise_for_status()
    return resp.json()

def collect(request_id, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = post("retrieve_future", {"request_id": request_id})
        if "error" in out:
            raise RuntimeError(out["error"])
        if out.get("type") != "try_again":
            return out
        time.sleep(0.2)
    raise TimeoutError(f"仍在等待 {request_id}")

model = post("get_server_capabilities", {})["supported_models"][0]["model_name"]
req = post("asample", {
    "base_model": model,
    "prompt": {"chunks": [{"type": "encoded_text", "tokens": [10, 11, 12]}]},
    "num_samples": 1,
    "sampling_params": {"max_tokens": 8, "temperature": 0}
})
print(collect(req["request_id"])["sequences"])
~~~

原始 HTTP 的训练顺序为 `create_session → create_model → retrieve_future → forward_backward → retrieve_future → optim_step → retrieve_future → save_weights → retrieve_future`。训练请求示例如下，`datum` 是第 3 节 JSON 解析出的一个对象，函数沿用上例。后台接入时保存每步的 `request_id`，轮询由一个消费者负责。已提交的 `optim_step` 遇到网络超时，直接重发可能多更新一次。

~~~python
sid = post("create_session", {})["session_id"]
created = post("create_model", {"session_id": sid, "model_seq_id": 0, "base_model": model})
mid = collect(created["request_id"])["model_id"]
job = post("forward_backward", {
    "model_id": mid, "seq_id": 1,
    "forward_backward_input": {"data": [datum], "loss_fn": "cross_entropy"},
})
print(collect(job["request_id"])["metrics"])
job = post("optim_step", {"model_id": mid, "seq_id": 2, "adam_params": {"learning_rate": 1e-5}})
print(collect(job["request_id"])["metrics"])
~~~

### 5.5　如何判定请求成功

| 现象 | 处理方式 |
|---|---|
| HTTP 503 | 等 `healthz.status` 成为 `ready`；若为 `error`，读 `message`。 |
| HTTP 422 | 查看 FastAPI 的 `detail` 字段，按 `loc` 定位嵌套层级、缺失字段或类型。异步任务内的校验失败也可能出现在 future 的 `error` 中。 |
| 会话或采样请求立即收到 400／404 | 检查模型公开名、传入的是 state URI 还是 sampler URI，并确认句柄属于当前服务器进程。 |
| 轮询收到 `try_again`，甚至提示 future 已完成 | 用同一 `request_id` 再领一次。有任务在等待过程中刚完成时，该实现会先提示重试。 |
| 轮询返回 HTTP 200，同时 JSON 内有 `error` | 业务操作失败，按 `error` 和 `category` 处理。旧 sampler 的版本冲突也可能以这种形式携带 409 原因；若需要最新策略就重新同步并取新句柄。 |
| `seen_status="seen_before"` ／ `"never_seen"` | 前者表示该结果已被领取，后者表示这个服务不认识该 ID；核对是否有多个轮询者、请求是否发往同一实例。 |

## 6　用这些接口实现 GRPO、OPD 和 SDFT

| 任务 | 客户端负责 | 服务端调用 |
|---|---|---|
| GRPO 风格 RL | 对同一题采 G 个回答；客户端按业务规则打分并计算组相对优势。回答 token 填入对应优势和行为策略 logprobs，prompt 用 0 权重。 | `sample → forward_backward("ppo" 或 "importance_sampling") → optim_step → 同步 rollout`。对 reference 求 KL 时另取基模标量 logprobs。 |
| 单／多 Teacher 的 on-policy 蒸馏 | 学生生成回答，Teacher 对同一轨迹打分，客户端对齐回答 token、组合 Teacher 信号并生成训练数据。 | 学生 `sample`，Teacher `compute_logprobs`，学生 `forward_backward → optim_step`。仓库 `opd_deepmath` 具体用 `dro`、`beta=0.05`；多 Teacher 按各自模型名建 sampler。 |
| 离线 Top-K 蒸馏 | 向 Teacher 取每个 prompt 位置的候选 ID 和对数概率；去除首个空位置，按目标位置对齐，选有效行归一化，将候选和权重按 `[N,K]` 组张量。Teacher、学生需共享兼容 token ID 或先显式处理词表对齐。 | `teacher.sample(..., topk_prompt_logprobs=K) → student.forward_backward(..., "cross_entropy") → optim_step`。仓库示例 K=20。 |
| 有上下文和无上下文的自蒸馏 | 客户端分别构造带示范的教师上下文和学生上下文，按回答边界对齐概率；需要核对两边 token 位置偏移。 | 仓库 `sdft_single_task` 示例 `topk=20`、`reverse=False`、`teacher_sync_every=1`，让教师使用当前 rollout；该配方不要求另起冻结 Teacher。 |

若要在 Z-Desktop 上做在线训练，单条 agent 轨迹可记录四组可复用数据：模板处理后的 prompt IDs、回答 IDs、生成时逐 token logprobs、业务评分与回答 mask。SFT 使用目标和权重；RL 还需要由评分得到 advantages；蒸馏再增加同一回答在 Teacher 下的标量或 Top-K 概率。HTTP 没有单独的 `/grpo`、`/opd` 或 `/sdft`。配置 `actor.yaml` 支持纯训练；`actor_rollout.yaml` 增加生成；`actor_rollout_ref.yaml` 增加 reference；专用冻结 Teacher 要看对应的高级部署 YAML。LoRA 模式能关闭 adapter 取基础模型标量分数。

## 7　随仓库附带的客户端 runner

`run_single_test.py` 是一次性测试入口，在 `finally` 里调用 `shutdown_server`，因此用它跑完测试（包括异常退出测试体）会向已连接的服务发送 `POST /api/v1/shutdown`。长期运行的服务可直接按第 5 节调用 SDK。

| 参数 | 默认值或可选值 |
|---|---|
| `--base-url` | `http://127.0.0.1:8000/` |
| `--api-key` | `tml-verl-tinker-local` |
| `--model-name` | `Qwen/Qwen3-1.7B`，部署其他模型时传真实公开名 |
| `--tokenizer-name-or-path` | 默认跟随 `--model-name`；可指定客户端本地目录 |
| `--test-name` | 默认 `sft_tulu3`；全部九项为 `sft_tulu3`、`sft_norobot`、`sft_norobot_no_rollout`、`topk_distillation`、`sdft_single_task`、`rl_gsm8k`、`sft_rl_gsm8k`、`opd_deepmath`、`opd_multi_teacher` |
| `--lite` | 默认关闭。选择较小的训练批次、采样和评估工作量，多个配方将训练上限设为 20 步。 |

配置定位：`TINKER_SERVER_MODEL_NAME` 是服务器公开名，`TINKER_SERVER_MODEL_PATH` 是服务端权重路径，客户端的 `TOKENIZER_DIR` 是上面示例自定义的本地路径。`TINKER_CHECKPOINT_DIR` 控制训练 checkpoint 目录，quick-start 默认 `/tmp/tinker-checkpoints`。`TINKER_SERVER_MAX_CONCURRENT_SAMPLES` 在 quick-start 默认 32；调度还受 vLLM 和显存配置制约。模型路径、GPU 数、TP、reference、Teacher 和 LoRA 应核对启动 YAML。当客户端、服务端用了不同环境时，要分别检查 SDK 版本。

## 源码定位

- [发布文章](https://verl-project.github.io/posts/2026-08-11-verl-tinker/)
- [固定提交下的服务 README、配置与锁文件](https://github.com/verl-project/verl-recipe/tree/0a9360bb366d8bd71e26fb86da1acf077da9e6d9/verl_tinker)
- [HTTP 路由、future 领取与 checkpoint 元数据](https://github.com/verl-project/verl-recipe/blob/0a9360bb366d8bd71e26fb86da1acf077da9e6d9/verl_tinker/src/verl_tinker/tinker_router.py)
- [loss、Adam、stop、Top-K 与权重同步](https://github.com/verl-project/verl-recipe/blob/0a9360bb366d8bd71e26fb86da1acf077da9e6d9/verl_tinker/src/verl_tinker/tinker_ops.py)
- [逐 token 数据与 Top-K 张量转换](https://github.com/verl-project/verl-recipe/blob/0a9360bb366d8bd71e26fb86da1acf077da9e6d9/verl_tinker/src/verl_tinker/data/datum_processing.py)
- [verl loss 的实际映射](https://github.com/verl-project/verl-recipe/blob/0a9360bb366d8bd71e26fb86da1acf077da9e6d9/verl_tinker/src/verl_tinker/backends/_loss.py)
- [checkpoint 恢复与清梯度](https://github.com/verl-project/verl-recipe/blob/0a9360bb366d8bd71e26fb86da1acf077da9e6d9/verl_tinker/src/verl_tinker/backends/colocated.py)
- [采样版本与资源选择](https://github.com/verl-project/verl-recipe/blob/0a9360bb366d8bd71e26fb86da1acf077da9e6d9/verl_tinker/src/verl_tinker/model_resource_manager.py)
- [SDK 示例、runner 和锁文件](https://github.com/verl-project/verl-recipe/tree/0a9360bb366d8bd71e26fb86da1acf077da9e6d9/verl_tinker/client_examples)
- [本地服务所用 Tinker 请求类型 0.19.0](https://pypi.org/project/tinker/0.19.0/) · [示例客户端 Tinker 0.24.0](https://pypi.org/project/tinker/0.24.0/) · [上游滚动 API 文档](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/)


