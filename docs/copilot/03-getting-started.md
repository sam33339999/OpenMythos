# 開始動手：安裝與第一個範例

> 這份文件會帶你從零開始，跑出第一個 OpenMythos 的程式碼。

---

## 你需要準備什麼？

- 一台電腦（Windows / Mac / Linux 都可以）
- Python 3.10 以上（建議 3.11）
- 網路連線（下載套件用）
- 不需要 GPU（跑小型測試用 CPU 就夠了）

---

## 第一步：安裝 Python

如果你還沒裝 Python，請前往官方網站下載：  
👉 https://www.python.org/downloads/

安裝時請勾選 **「Add Python to PATH」**（很重要！否則終端機找不到 Python）。

安裝完後，打開終端機（Windows 叫「命令提示字元」或「PowerShell」），確認安裝成功：

```bash
python --version
# 應該顯示 Python 3.10.x 或更高版本
```

---

## 第二步：建立虛擬環境（建議）

虛擬環境可以讓每個專案的套件互不干擾，養成好習慣：

```bash
# 建立虛擬環境（只需要做一次）
python -m venv mythos-env

# 啟動虛擬環境
# Windows:
mythos-env\Scripts\activate
# Mac / Linux:
source mythos-env/bin/activate

# 啟動後，你的終端機提示符前面會多出 (mythos-env)
```

---

## 第三步：安裝 OpenMythos

```bash
pip install open-mythos
```

這會自動安裝 OpenMythos 和它需要的套件（包括 PyTorch）。

> 💡 如果你使用 `uv`（更快的套件管理工具），可以用：
> ```bash
> uv pip install open-mythos
> ```

---

## 第四步：跑第一個程式

建立一個新檔案，命名為 `hello_mythos.py`，貼上以下程式碼：

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

# 建立一個很小的模型設定（適合測試用，不需要 GPU）
cfg = MythosConfig(
    vocab_size=1000,    # 詞彙表大小（訓練時通常是 32000）
    dim=256,            # 模型的「思考寬度」
    n_heads=4,          # 注意力頭的數量
    n_kv_heads=2,       # KV 注意力頭（GQA 模式）
    max_seq_len=128,    # 最長輸入長度
    max_loop_iters=4,   # 最多循環幾次
    prelude_layers=1,   # 前奏層數
    coda_layers=1,      # 尾聲層數
    n_experts=8,        # 專家數量（MoE）
    n_shared_experts=1, # 共用專家數量
    n_experts_per_tok=2,# 每個詞用幾個專家
    expert_dim=64,      # 每個專家的維度
    lora_rank=4,        # LoRA 適配器的秩
    attn_type="gqa",    # 注意力類型（gqa 比較簡單）
)

# 建立模型
model = OpenMythos(cfg)

# 查看模型有多少參數
total_params = sum(p.numel() for p in model.parameters())
print(f"模型參數數量：{total_params:,}")

# 建立一個假的輸入（batch_size=1, sequence_length=16）
input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
print(f"輸入形狀：{input_ids.shape}")

# 跑一次前向傳播（模型「思考」一次）
logits = model(input_ids, n_loops=4)
print(f"輸出 logits 形狀：{logits.shape}")
# 輸出應該是 (1, 16, 1000)，代表每個位置對 1000 個詞的預測機率

print("✅ 成功！你的第一個 OpenMythos 模型跑起來了！")
```

執行它：

```bash
python hello_mythos.py
```

你應該會看到類似這樣的輸出：

```
模型參數數量：2,345,678
輸入形狀：torch.Size([1, 16])
輸出 logits 形狀：torch.Size([1, 16, 1000])
✅ 成功！你的第一個 OpenMythos 模型跑起來了！
```

---

## 第五步：試試看文字生成

雖然沒有訓練過的模型只會輸出隨機結果，但我們可以看看生成機制如何運作：

```python
import torch
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
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    lora_rank=4,
    attn_type="gqa",
)

model = OpenMythos(cfg)
model.eval()  # 切換到推理模式

# 假設這是一段「提示詞」（實際上是隨機 token ID）
prompt = torch.randint(0, cfg.vocab_size, (1, 8))
print(f"提示詞長度：{prompt.shape[1]} 個 token")

# 生成 10 個新 token
output = model.generate(
    prompt,
    max_new_tokens=10,  # 生成幾個新詞
    n_loops=4,          # 每步用幾次循環（越多越深思熟慮）
    temperature=0.8,    # 溫度：越低越保守，越高越多樣
    top_k=50,           # 只從最高機率的 50 個詞中選
)

print(f"輸出長度：{output.shape[1]} 個 token（8 個原始 + 10 個生成）")
print(f"生成的 token IDs：{output[0, 8:].tolist()}")
```

---

## 常見問題

### Q: 出現 `ModuleNotFoundError: No module named 'torch'` 怎麼辦？

確認你的虛擬環境已啟動，然後重新安裝：

```bash
pip install torch
pip install open-mythos
```

### Q: 速度很慢怎麼辦？

沒有 GPU 的話，CPU 確實比較慢。小型模型（dim=256）通常幾秒內就能跑完。如果你用預設的大設定，會慢很多——建議測試時使用本文件的小型設定。

### Q: 我能改變循環次數嗎？

可以！`n_loops` 就是控制循環次數的參數：

```python
# 快速但思考較淺
logits = model(input_ids, n_loops=4)

# 慢一點但思考較深（可以超過訓練時設定的 max_loop_iters）
logits = model(input_ids, n_loops=32)
```

---

## 你現在學到了什麼

- ✅ 安裝了 OpenMythos
- ✅ 建立了第一個模型
- ✅ 跑了前向傳播，理解了輸出的形狀
- ✅ 使用了 `generate` 方法生成新 token
- ✅ 知道怎麼調整循環深度

---

下一步：[→ 訓練你的第一個模型](04-train-your-first-model.md)
