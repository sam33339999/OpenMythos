# OpenMythos — 繁體中文說明文件

<p align="left">
  <a href="https://pypi.org/project/open-mythos/" target="_blank">
    <picture>
      <source srcset="https://img.shields.io/pypi/v/open-mythos?style=for-the-badge&color=3670A0" media="(prefers-color-scheme: dark)">
      <img alt="Version" src="https://img.shields.io/pypi/v/open-mythos?style=for-the-badge&color=3670A0">
    </picture>
  </a>
  <a href="https://twitter.com/kyegomezb/">
    <picture>
      <source srcset="https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" media="(prefers-color-scheme: dark)">
      <img src="https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter">
    </picture>
  </a>
  <a href="https://discord.gg/3keGBK9Pvr" target="_blank">
    <picture>
      <source srcset="https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord&logoColor=white" media="(prefers-color-scheme: dark)">
      <img alt="Discord" src="https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord&logoColor=white">
    </picture>
  </a>
  <a href="https://pytorch.org" target="_blank">
    <picture>
      <source srcset="https://img.shields.io/badge/PyTorch-Implemented-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" media="(prefers-color-scheme: dark)">
      <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Implemented-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
    </picture>
  </a>
</p>

> **免責聲明：** OpenMythos 是一個獨立的、社群驅動的理論性重建專案，僅基於公開可用的研究資料與推測。本專案與 Anthropic 或其任何專有系統無關，未獲授權，亦無任何關聯。

---

## 📖 這個專案在幹嘛？

**OpenMythos** 是一個開源的理論性實作，試圖重建 Anthropic 的 **Claude Mythos** 模型架構。它的核心假設是：Claude Mythos 使用了一種稱為 **循環深度 Transformer（Recurrent-Depth Transformer，RDT）** 的架構，而不是傳統的「疊很多層」的方式。

### 一句話解釋

> 傳統 LLM 是把 100 層 Transformer 堆起來跑一次；OpenMythos 是把 **同一組 Transformer** 反覆跑多次（迴圈），每次迭代都在潛在空間裡做更深層的推理——這叫做 **隱式思維鏈（Implicit Chain-of-Thought）**，完全在模型內部完成，不需要輸出任何中間 token。

---

## 🏗️ 架構總覽

```
輸入 token IDs  (B, T)
      ↓
 [Embedding]        token → dim 維向量
      ↓
 [Prelude]          prelude_layers 層標準 Transformer（只跑一次）
      ↓
 [Recurrent Block]  一個 Transformer 迴圈跑 T 次
   ↑__________↓    h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
      ↓
 [Coda]             coda_layers 層標準 Transformer（只跑一次）
      ↓
 [RMSNorm → LM head]
      ↓
輸出 logits  (B, T, vocab_size)
```

### 三個主要階段

| 階段 | 說明 |
|---|---|
| **Prelude（序幕）** | 標準 Transformer 層，負責將輸入 token 編碼成隱藏表示 `e`，只跑一次 |
| **Recurrent Block（循環塊）** | 核心！同一組 Transformer 迴圈跑 `max_loop_iters` 次，每次迭代更新隱藏狀態 `h`，並持續注入原始編碼 `e` 防止偏移 |
| **Coda（尾聲）** | 標準 Transformer 層，負責將最終隱藏狀態解碼成 token 預測，只跑一次 |

### 關鍵技術元件

| 技術 | 作用 |
|---|---|
| **LTI Injection（線性時不變注入）** | 穩定的循環更新規則，透過 ZOH 離散化確保訓練穩定（光譜半徑 < 1） |
| **ACT Halting（自適應計算時間）** | 讓簡單 token 提早停止，複雜 token 跑滿迴圈，節省計算資源 |
| **MoE FFN（混合專家前饋網路）** | 循環塊內使用稀疏 MoE，每個 token 只激活約 6.25% 的專家參數 |
| **MLA / GQA Attention** | 可切換的注意力機制：MLA 的 KV cache 比標準注意力小 10–20 倍 |
| **Depth-wise LoRA** | 每個迴圈迭代有一個低秩適配器，讓同一組權重在不同深度有不同行為 |

---

## 🚀 快速開始（新手指南）

### 第一步：確認 Python 環境

OpenMythos 需要 Python 3.10 或以上版本。

```bash
python --version
# 應輸出 Python 3.10.x 或更高
```

### 第二步：安裝套件

```bash
# 使用 pip 安裝（推薦）
pip install open-mythos

# 或使用 uv（更快）
uv pip install open-mythos
```

### 第三步：驗證安裝

```python
from open_mythos.main import OpenMythos, MythosConfig

cfg = MythosConfig(
    vocab_size=1000,
    dim=256,
    n_heads=4,
    n_kv_heads=2,
    max_seq_len=128,
    max_loop_iters=4,
    prelude_layers=1,
    coda_layers=1,
    attn_type="gqa",
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    lora_rank=4,
)

model = OpenMythos(cfg)
total = sum(p.numel() for p in model.parameters())
print(f"模型參數量：{total:,}")
```

---

## 💻 不同硬體架構啟動方式

### 方式一：CUDA（NVIDIA GPU）

這是訓練大型模型的最佳選擇。

**環境需求：**
- NVIDIA GPU（建議 RTX 3090、A100、H100 等）
- CUDA 12.x
- PyTorch with CUDA support

**安裝 PyTorch（CUDA 版）：**

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**使用範例：**

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

# 確認 CUDA 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")
print(f"GPU 名稱：{torch.cuda.get_device_name(0)}")

cfg = MythosConfig(
    vocab_size=32000,
    dim=2048,
    n_heads=16,
    n_kv_heads=4,
    max_loop_iters=16,
    attn_type="mla",
)

# 將模型移到 CUDA
model = OpenMythos(cfg).to(device)
print(f"參數量：{sum(p.numel() for p in model.parameters()):,}")

# 前向推理
import torch
ids = torch.randint(0, 32000, (1, 64)).to(device)
logits = model(ids, n_loops=16)
print(f"輸出 shape：{logits.shape}")  # (1, 64, 32000)

# 文字生成
output = model.generate(
    ids,
    max_new_tokens=128,
    n_loops=16,
    temperature=0.8,
    top_k=40,
)
print(f"生成 shape：{output.shape}")
```

**多 GPU 訓練（DDP）：**

```bash
# 自動偵測 GPU 數量並啟動
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") \
    training/3b_fine_web_edu.py
```

**混合精度訓練（節省記憶體）：**

```python
import torch
from torch.cuda.amp import autocast, GradScaler
from open_mythos.main import OpenMythos, MythosConfig

model = OpenMythos(MythosConfig()).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = GradScaler()  # 用於 float16 的梯度縮放

ids    = torch.randint(0, 32000, (2, 512)).cuda()
labels = torch.randint(0, 32000, (2, 512)).cuda()

with autocast(dtype=torch.bfloat16):   # A100/H100 用 bfloat16
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, 32000), labels.view(-1)
    )

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

### 方式二：Apple Silicon MPS（M1 / M2 / M3 / M4）

PyTorch 透過 **MPS（Metal Performance Shaders）** 後端支援 Apple Silicon GPU。

**環境需求：**
- macOS 12.3 以上
- Apple Silicon Mac（M1、M2、M3、M4 系列）
- PyTorch 2.0+（含 MPS 支援）

**安裝 PyTorch（macOS 版）：**

```bash
pip install torch torchvision torchaudio
```

**使用範例：**

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

# 確認 MPS 可用
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple Silicon MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ 使用 CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️ 使用 CPU（速度較慢）")

# 建議 Apple Silicon 使用較小的模型配置
cfg = MythosConfig(
    vocab_size=32000,
    dim=1024,
    n_heads=8,
    n_kv_heads=2,
    max_seq_len=2048,
    max_loop_iters=8,
    prelude_layers=2,
    coda_layers=2,
    attn_type="gqa",   # MPS 上 GQA 相容性更好
    n_experts=16,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=256,
    lora_rank=8,
)

model = OpenMythos(cfg).to(device)
print(f"參數量：{sum(p.numel() for p in model.parameters()):,}")

ids = torch.randint(0, 32000, (1, 32)).to(device)
with torch.no_grad():
    output = model.generate(ids, max_new_tokens=32, n_loops=8)
print(f"生成 shape：{output.shape}")
```

> **提示：** MPS 後端的功能仍在持續完善中。若遇到算子不支援的錯誤，可嘗試退到 CPU：`device = torch.device("cpu")`。

---

### 方式三：MLX（Apple Silicon 原生框架）

**MLX** 是 Apple 專為 Apple Silicon 設計的機器學習框架，比 PyTorch MPS 更原生、效率更高，但 **OpenMythos 目前為 PyTorch 實作**，沒有直接的 MLX 後端。

若你想在 Apple Silicon 上獲得最佳效能，有以下選項：

#### 選項 A：使用 PyTorch MPS 後端（推薦，無需修改程式碼）

如上方「方式二」所述，直接使用 `.to("mps")` 即可。

#### 選項 B：安裝 MLX 並並行使用

```bash
pip install mlx mlx-lm
```

MLX 目前主要支援載入 Hugging Face 上已有的模型格式（如 `mlx-community` 上的量化模型）。若你想完整移植 OpenMythos 到 MLX，需要手動將 PyTorch 的各個模組轉換為 MLX 對應的 API。

#### 選項 C：CPU 模式（任何設備均可使用）

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

# 使用預設小型配置在 CPU 上執行
cfg = MythosConfig(
    vocab_size=8192,
    dim=256,
    n_heads=4,
    n_kv_heads=2,
    max_seq_len=512,
    max_loop_iters=4,
    prelude_layers=1,
    coda_layers=1,
    attn_type="gqa",
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    lora_rank=4,
)

model = OpenMythos(cfg)  # 預設在 CPU
ids = torch.randint(0, 8192, (1, 16))
output = model.generate(ids, max_new_tokens=16, n_loops=4)
print(f"生成完成：{output.shape}")
```

---

### 各後端比較

| 後端 | 適用硬體 | 速度 | 支援程度 | 備註 |
|---|---|---|---|---|
| **CUDA** | NVIDIA GPU | ⚡⚡⚡⚡ | ✅ 完整支援 | 訓練大模型首選 |
| **MPS** | Apple Silicon (M 系列) | ⚡⚡⚡ | ✅ 支援 | 部分算子可能有限制 |
| **MLX** | Apple Silicon (M 系列) | ⚡⚡⚡⚡ | ⚠️ 需手動移植 | 原生更快，但需自行轉換 |
| **CPU** | 任何設備 | ⚡ | ✅ 完整支援 | 僅適合測試小模型 |

---

## 📦 模型變體（預設配置）

專案提供從 1B 到 1T 參數量的預設配置，方便直接使用：

```python
from open_mythos import (
    mythos_1b,
    mythos_3b,
    mythos_10b,
    mythos_50b,
    mythos_100b,
    mythos_500b,
    mythos_1t,
    OpenMythos,
)

# 以 1B 模型為例
cfg = mythos_1b()
model = OpenMythos(cfg)
print(f"參數量：{sum(p.numel() for p in model.parameters()):,}")
```

| 變體 | 隱藏維度 | 專家數 | 迴圈次數 | 上下文長度 |
|---|---|---|---|---|
| `mythos_1b` | 2048 | 64 | 16 | 4k |
| `mythos_3b` | 3072 | 64 | 16 | 4k |
| `mythos_10b` | 4096 | 128 | 24 | 8k |
| `mythos_50b` | 6144 | 256 | 32 | 8k |
| `mythos_100b` | 8192 | 256 | 32 | 1M |
| `mythos_500b` | 12288 | 512 | 48 | 1M |
| `mythos_1t` | 16384 | 512 | 64 | 1M |

> **提示：** 較小的模型（1B、3B）適合研究與本地實驗；較大的模型（10B+）需要多 GPU 或雲端算力。

---

## 🎓 完整使用範例

### 範例一：MLA 注意力（預設，KV cache 最小）

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

# MLA（Multi-Latent Attention）配置
cfg = MythosConfig(
    vocab_size=1000,
    dim=256,
    n_heads=8,
    n_kv_heads=8,
    max_seq_len=128,
    max_loop_iters=4,
    prelude_layers=1,
    coda_layers=1,
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    lora_rank=8,
    attn_type="mla",      # MLA：KV cache 比標準注意力小 10-20 倍
    kv_lora_rank=32,
    q_lora_rank=64,
    qk_rope_head_dim=16,
    qk_nope_head_dim=16,
    v_head_dim=16,
)

model = OpenMythos(cfg)
print(f"[MLA] 參數量：{sum(p.numel() for p in model.parameters()):,}")

ids = torch.randint(0, cfg.vocab_size, (2, 16))
logits = model(ids, n_loops=4)
print(f"[MLA] 輸出 shape：{logits.shape}")

output = model.generate(ids, max_new_tokens=8, n_loops=8)
print(f"[MLA] 生成 shape：{output.shape}")
```

### 範例二：GQA 注意力（較輕量）

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

# GQA（Grouped Query Attention）配置
cfg = MythosConfig(
    vocab_size=1000,
    dim=256,
    n_heads=8,
    n_kv_heads=2,           # 4 個 Q 頭共享 1 個 KV 頭
    max_seq_len=128,
    max_loop_iters=4,
    prelude_layers=1,
    coda_layers=1,
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    lora_rank=8,
    attn_type="gqa",
)

model = OpenMythos(cfg)
print(f"[GQA] 參數量：{sum(p.numel() for p in model.parameters()):,}")
```

### 範例三：推理時加深迴圈（深度外推）

這是 RDT 架構最強大的特性之一：訓練時用 N 個迴圈，推理時可以用 N+k 個迴圈解決更難的問題！

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

model = OpenMythos(MythosConfig()).eval()

# 假設已用 16 個迴圈訓練完成...
ids = torch.randint(0, 32000, (1, 32))

# 推理時可以使用更多迴圈解決更複雜的問題
with torch.no_grad():
    logits_normal = model(ids, n_loops=16)   # 標準深度
    logits_deep   = model(ids, n_loops=32)   # 更深的推理

print(f"標準推理 shape：{logits_normal.shape}")
print(f"深度外推 shape：{logits_deep.shape}")
```

### 範例四：訓練迴圈

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

model = OpenMythos(MythosConfig()).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 模擬一個訓練 step
input_ids = torch.randint(0, 32000, (2, 512)).cuda()
labels    = torch.randint(0, 32000, (2, 512)).cuda()

logits = model(input_ids)                    # (2, 512, 32000)
loss = torch.nn.functional.cross_entropy(
    logits.view(-1, 32000),
    labels.view(-1),
)
loss.backward()
optimizer.step()
optimizer.zero_grad()
print(f"Loss: {loss.item():.4f}")

# 確認穩定性：光譜半徑必須 < 1
A = model.recurrent.injection.get_A()
print(f"光譜半徑 ρ(A) 最大值：{A.max().item():.4f}（必須 < 1）")
```

---

## 🏋️ 訓練大模型

### 單 GPU 訓練

```bash
python training/3b_fine_web_edu.py
```

### 多 GPU 訓練（DDP）

```bash
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") \
    training/3b_fine_web_edu.py
```

### 訓練配置說明

| 設定 | 預設值 | 說明 |
|---|---|---|
| 最佳化器 | AdamW | 標準 LLM 訓練最佳化器 |
| 資料集 | FineWeb-Edu `sample-10BT` | 高品質教育網頁文字，1.3T tokens |
| Tokenizer | `openai/gpt-oss-20b` | 透過 `MythosTokenizer` 封裝 |
| 精度 | bfloat16（A100/H100）| 舊 GPU 用 float16 + GradScaler |
| 學習率排程 | 線性暖身 2000 步 → Cosine 衰減 | — |

### 推薦訓練資料集

| 資料集 | Token 數 | 授權 | 用途 |
|---|---|---|---|
| FineWeb-Edu | 1.3T | Apache 2.0 | 主要預訓練 |
| OpenHermes 2.5 | ~1M 樣本 | Apache 2.0 | 指令微調（~5% 混合） |
| OpenWebMath | 14.7B | ODC-By | 數學/推理能力強化 |

---

## 🧠 為什麼這個架構很重要？

### 1. 系統性泛化（Systematic Generalization）

傳統 Transformer 難以組合它從未見過的知識。循環 Transformer 透過三階段的「頓悟（Grokking）」過程學會了這個能力：
1. 記憶訓練資料
2. 在分布內泛化
3. **突然**能夠在分布外（OOD）組合新知識

### 2. 深度外推（Depth Extrapolation）

訓練時用 5 步推理鏈，推理時可以跑 10 步——Vanilla Transformer 做不到，但循環架構可以。這直接解釋了為什麼 Mythos 能處理複雜的多步數學、長期規劃和多層次論證。

### 3. 潛在思維（Latent Thoughts）

每個迴圈迭代等價於一步思維鏈（Chain-of-Thought），但在**連續潛在空間**中操作而非 token 空間。這已被 Saunshi et al.（2025）正式證明。

### 4. 參數效率

使用 k 層跑 L 次迴圈，等效於 kL 層的非迴圈模型的品質，但參數量只有 k 層。
> 在 770M 參數下，循環模型達到了同等資料量下 1.3B 固定深度 Transformer 的品質。

### 5. 訓練穩定性

透過 **LTI Injection** 機制，確保光譜半徑 ρ(A) < 1，從根本上解決了循環訓練的爆炸問題。

---

## 📚 核心設計參數說明

### `MythosConfig` 主要欄位

| 欄位 | 預設值 | 說明 |
|---|---|---|
| `vocab_size` | 32000 | 詞彙表大小 |
| `dim` | 2048 | 模型隱藏維度（殘差流寬度） |
| `n_heads` | 16 | 查詢注意力頭數量 |
| `n_kv_heads` | 4 | KV 頭數量（GQA 用） |
| `max_seq_len` | 4096 | 最大序列長度 |
| `max_loop_iters` | 16 | 推理時的預設迴圈深度 |
| `prelude_layers` | 2 | Prelude 層數 |
| `coda_layers` | 2 | Coda 層數 |
| `attn_type` | `"mla"` | 注意力類型：`"mla"` 或 `"gqa"` |
| `n_experts` | 64 | MoE 總專家數 |
| `n_experts_per_tok` | 4 | 每個 token 路由的專家數 |
| `act_threshold` | 0.99 | ACT 累積停止閾值 |
| `lora_rank` | 16 | 深度 LoRA 的秩 |

---

## 📁 專案結構

```
OpenMythos/
├── open_mythos/
│   ├── main.py          # 核心模型實作（OpenMythos、MythosConfig 等）
│   ├── variants.py      # 預設模型變體（mythos_1b ~ mythos_1t）
│   ├── tokenizer.py     # MythosTokenizer 封裝
│   └── moda.py          # 額外模組
├── training/
│   └── 3b_fine_web_edu.py  # 3B 模型訓練腳本
├── docs/
│   ├── open_mythos.md   # 完整 API 文件
│   └── datasets.md      # 訓練資料集建議
├── example.py           # 快速使用範例
├── variants_example.py  # 模型變體範例
└── README.md            # 英文說明文件
```

---

## 🔧 常見問題（FAQ）

**Q：沒有 GPU 可以用嗎？**

可以，直接不呼叫 `.to(device)` 或使用 `device = "cpu"` 即可。但建議只用小型配置（`dim=256`、`max_loop_iters=4`）測試，否則速度會非常慢。

**Q：和 llama.cpp / Ollama 相容嗎？**

目前不相容。OpenMythos 是 PyTorch 實作，需要在 Python 環境中執行。若要移植至 llama.cpp，需要自行轉換模型格式。

**Q：要怎麼載入已訓練的權重？**

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

cfg = MythosConfig(...)
model = OpenMythos(cfg)
model.load_state_dict(torch.load("path/to/checkpoint.pt"))
model.eval()
```

**Q：推理時多少個迴圈比較好？**

一般而言：
- 簡單問題：4–8 個迴圈
- 中等難度：8–16 個迴圈（訓練預設值）
- 複雜推理：16–32 個迴圈（深度外推）

**Q：MLA 和 GQA 要選哪個？**

- **MLA**：KV cache 更小，適合長序列和生產環境
- **GQA**：實作更簡單，在 Apple Silicon MPS 上相容性更好

---

## 📖 延伸閱讀

| 論文 | 說明 |
|---|---|
| [Loop, Think, & Generalize (2025)](https://arxiv.org/pdf/2604.07822) | 循環深度 Transformer 的推理能力 |
| [Parcae — Scaling Laws (Prairie et al., 2026)](https://arxiv.org/abs/2604.12946) | 穩定循環語言模型的縮放律 |
| [Reasoning with Latent Thoughts (Saunshi et al., 2025)](https://arxiv.org/abs/2502.17416) | 循環 Transformer 的潛在思維推理 |
| [DeepSeek-V2 MLA (2024)](https://arxiv.org/abs/2405.04434) | 多潛在注意力機制 |
| [DeepSeekMoE (Dai et al., 2024)](https://arxiv.org/abs/2401.06066) | 細粒度混合專家架構 |
| [Relaxed Recursive Transformers (Bae et al., 2024)](https://arxiv.org/pdf/2410.20672) | 深度 LoRA 參數共享 |

---

## 引用

如果你在研究中使用了 OpenMythos，請引用：

```bibtex
@software{gomez2026openmythos,
  author    = {Kye Gomez},
  title     = {OpenMythos: A Theoretical Reconstruction of the Claude Mythos Architecture},
  year      = {2026},
  url       = {https://github.com/kyegomez/OpenMythos},
  note      = {Recurrent-Depth Transformer with MoE, MLA, LTI-stable injection, and ACT halting}
}
```

---

## 授權

MIT License — Copyright (c) 2026 Kye Gomez。詳見 [`LICENSE`](LICENSE)。
