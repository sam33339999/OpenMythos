# 訓練你的第一個模型

> 這份文件會帶你訓練一個玩具版的 OpenMythos，讓你看到模型從「一無所知」到「開始學習」的過程。

---

## 先搞懂「訓練」是什麼

訓練一個語言模型，本質上就是讓它做一件事：

> **「給你前面的文字，預測下一個字是什麼」**

舉例：
- 輸入：「今天天氣很」
- 正確答案：「好」
- 模型猜：「壞」
- 差異（Loss）太大 → 調整參數 → 下次猜得更準

重複上面的過程幾千次、幾萬次，模型就漸漸「學會」了語言。

---

## 訓練所需要的三個東西

1. **模型**：OpenMythos（架構已經準備好了）
2. **資料**：要學習的文字（訓練時通常用大量網路文章；這裡用一個小例子）
3. **最佳化器**：決定每次怎麼調整參數（最常用 AdamW）

---

## 動手訓練一個玩具模型

建立一個新檔案 `train_toy.py`：

```python
import torch
import torch.nn.functional as F
from open_mythos.main import OpenMythos, MythosConfig

# ─────────────────────────────────────────
# 1. 準備一個超小的設定（在 CPU 上也能快速跑）
# ─────────────────────────────────────────
cfg = MythosConfig(
    vocab_size=256,     # 只學 256 種 token（ASCII 字元）
    dim=128,            # 模型寬度縮到最小
    n_heads=4,
    n_kv_heads=2,
    max_seq_len=64,
    max_loop_iters=4,
    prelude_layers=1,
    coda_layers=1,
    n_experts=4,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=32,
    lora_rank=4,
    attn_type="gqa",
)

model = OpenMythos(cfg)
print(f"模型參數數量：{sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────────────────────────
# 2. 準備訓練資料（我們用一段重複的字串模擬）
# ─────────────────────────────────────────
# 把文字轉換成 token ID（這裡直接用 ASCII 碼）
text = "hello world " * 100  # 重複 100 次，讓模型有東西學
token_ids = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)

print(f"訓練資料長度：{len(token_ids)} 個 token")

# ─────────────────────────────────────────
# 3. 定義最佳化器
# ─────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ─────────────────────────────────────────
# 4. 開始訓練！
# ─────────────────────────────────────────
seq_len = 32   # 每次訓練用多長的序列
batch_size = 4 # 每次同時訓練幾個序列
n_steps = 200  # 訓練幾步

model.train()  # 切換到訓練模式

for step in range(n_steps):
    # 隨機從資料中取出一批序列
    # 取 input（前 seq_len 個）和 label（後移一位的 seq_len 個）
    starts = torch.randint(0, len(token_ids) - seq_len - 1, (batch_size,))
    
    x = torch.stack([token_ids[s : s + seq_len] for s in starts])       # (B, T)
    y = torch.stack([token_ids[s + 1 : s + seq_len + 1] for s in starts])  # (B, T)

    # 前向傳播
    logits = model(x, n_loops=4)  # (B, T, vocab_size)

    # 計算 Loss（預測錯誤的程度）
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

    # 反向傳播（計算梯度）
    optimizer.zero_grad()
    loss.backward()
    
    # 更新參數
    optimizer.step()

    # 每 20 步印一次訓練進度
    if (step + 1) % 20 == 0:
        print(f"Step {step + 1:3d} / {n_steps} | Loss: {loss.item():.4f}")

print("\n✅ 訓練完成！")
```

執行：

```bash
python train_toy.py
```

你應該會看到 Loss 逐漸下降，像這樣：

```
模型參數數量：123,456
訓練資料長度：1200 個 token
Step  20 / 200 | Loss: 5.2341
Step  40 / 200 | Loss: 4.8102
Step  60 / 200 | Loss: 4.3567
Step  80 / 200 | Loss: 3.9123
Step 100 / 200 | Loss: 3.5874
...
Step 200 / 200 | Loss: 2.1234
✅ 訓練完成！
```

**Loss 下降 = 模型在學習。** 這就是訓練成功的標誌。

---

## 訓練完之後：看看模型生成什麼

在 `train_toy.py` 最後加上這段：

```python
# ─────────────────────────────────────────
# 5. 訓練後，試著生成一些輸出
# ─────────────────────────────────────────
model.eval()

# 用 "hell" 當作提示（ASCII 碼）
prompt_text = "hell"
prompt_ids = torch.tensor([[ord(c) % 256 for c in prompt_text]], dtype=torch.long)

with torch.no_grad():
    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=20,
        n_loops=4,
        temperature=0.8,
        top_k=20,
    )

# 把 token ID 轉回文字
output_text = "".join([chr(i) for i in output_ids[0].tolist()])
print(f"\n提示詞：{prompt_text!r}")
print(f"生成結果：{output_text!r}")
```

如果訓練成功，模型應該會傾向生成 `"hello world "` 這樣的字串，因為它學到了這個重複的模式。

---

## 儲存和載入模型

訓練好的模型可以存起來，下次不用重新訓練：

```python
import torch

# 儲存
torch.save(model.state_dict(), "my_toy_model.pt")
print("模型已儲存到 my_toy_model.pt")

# 載入（下次用）
model2 = OpenMythos(cfg)  # 先建立一個空的模型
model2.load_state_dict(torch.load("my_toy_model.pt"))
model2.eval()
print("模型已成功載入！")
```

---

## 增加循環深度：測試時更聰明

OpenMythos 有個特別的能力：**訓練時用 4 次循環，測試時可以用更多次，解更難的問題**：

```python
# 訓練時：n_loops=4
# 測試時：n_loops=8（更深的思考）
logits_deep = model(x, n_loops=8)
```

這叫做「深度外推（depth extrapolation）」——是這個架構的核心優勢之一。

---

## 下一步：訓練真正的模型

玩具訓練只是起點。如果你想訓練一個真正的語言模型：

1. **選資料集**：參考 [docs/datasets.md](../datasets.md)，推薦從 `FineWeb-Edu sample-10BT` 開始
2. **選模型大小**：參考 README 的變體表格（`mythos_1b` 是 1B 參數的版本）
3. **執行訓練腳本**：

```bash
# 單 GPU
python training/3b_fine_web_edu.py

# 多 GPU（自動偵測）
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") \
    training/3b_fine_web_edu.py
```

> ⚠️ **注意**：訓練 1B+ 的模型需要高端 GPU（A100 或 H100）和大量時間。如果只是學習，請用上面的玩具範例。

---

## 你現在學到了什麼

- ✅ 理解「訓練 = 讓模型學會預測下一個字」
- ✅ 自己訓練了一個玩具模型，看到 Loss 下降
- ✅ 試過用訓練好的模型生成文字
- ✅ 知道怎麼儲存和載入模型
- ✅ 了解「深度外推」是什麼

---

## 學習資源

| 資源 | 說明 |
|---|---|
| [docs/open_mythos.md](../open_mythos.md) | 完整 API 文件（適合進階使用） |
| [docs/datasets.md](../datasets.md) | 訓練用資料集推薦 |
| [example.py](../../example.py) | 更多使用範例 |
| [training/3b_fine_web_edu.py](../../training/3b_fine_web_edu.py) | 完整 3B 訓練腳本 |

---

**恭喜你！** 你已經完成了完整的新手入門流程。如果你有任何問題，歡迎到專案的 [Discord 社群](https://discord.gg/3keGBK9Pvr) 提問！
