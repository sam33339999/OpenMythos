# 08 — 迴圈索引嵌入與深度 LoRA 適配

> 承接第 07 篇。上一篇的 MoE 讓「不同 token 走不同專家」；本文處理另一個對稱問題：
> **同一組共享權重，如何在「不同的迴圈迭代」做不同的事？**
> 若沒有任何訊號區分迴圈深度，權重被迫同時勝任「早期模式匹配」與「晚期精煉」——太緊的約束。
> OpenMythos 用兩個互補機制緩解：`loop_index_embedding`（無參數訊號）與 `LoRAAdapter`（少量參數微調）。

## 1. 問題：為什麼需要區分迴圈深度

設想 RecurrentBlock 的 TransformerBlock 跑 16 次（`max_loop_iters=16`）。若每次輸入完全相同，
權重只能學到「一個對所有深度都尚可的折衷」。`README.md:316-321` 把這稱為「tight constraint」，
並提出類比：

> 就像 RoPE 讓同一個注意力頭在不同序列位置表現不同，一個 **RoPE-like 的迴圈索引嵌入**，
> 與輸入一起在每步注入，就能讓同一組參數在不同迴圈迭代實現功能上不同的運算。

OpenMythos 實作了這個想法的兩個層次。

## 2. 機制一：`loop_index_embedding`（無參數）

`loop_index_embedding`（`main.py:541-570`）在隱藏狀態的前 `loop_dim` 個通道注入正弦訊號：

```python
def loop_index_embedding(h, loop_t, loop_dim, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, loop_dim, 2, ...) / loop_dim))
    angles = loop_t * freqs                  # (loop_dim//2,)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], ...)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)
```

這與 RoPE 在序列位置的邏輯同構，只是把「序列位置 m」換成「迴圈深度 t」。特點：

- **無可學參數**：純函數，靠 `theta`（固定 10000）決定頻率。
- **只動前 `loop_dim` 通道**：RecurrentBlock 設 `loop_dim = dim // 8`（`main.py:821-823`），
  其餘通道不變。測試 `test_only_first_dims_modified`（`tests/test_main.py:395-400`）驗證這點。
- **不同迭代產生不同輸出**：`test_different_iterations_differ`（`tests/test_main.py:389-393`）。

在 RecurrentBlock 內的呼叫點（`main.py:858`）：

```python
h_loop = loop_index_embedding(h, t, self.loop_dim)
combined = self.norm(h_loop + e)   # 再加凍結 e、正規化，才進 TransformerBlock
```

注意它是「先注入迴圈索引、再加 e、再 norm」——三個訊號在進注意力前就疊好。

## 3. 機制二：`LoRAAdapter`（少量參數）

`LoRAAdapter`（`main.py:578-619`）源自 Relaxed Recursive Transformers（Bae et al., 2024）。
它介於「純權重共享」與「每層獨立權重」之間（`main.py:580-589`）：

```
delta(x, t) = (down(x) * scale[t]) @ B
```

三個組件（`main.py:599-601`）：

| 組件 | 形狀 | 共享性 |
|------|------|--------|
| `down` | `Linear(dim, rank)` | 跨所有迴圈共享 |
| `B` | `(rank, dim)` 參數 | 跨所有迴圈共享 |
| `scale` | `Embedding(max_loops, rank)` | **每個迴圈一個**元素級尺度向量 |

也就是：共享的低秩 down/up 投影，加上一個「按迴圈深度查表」的逐元素尺度。
總參數開銷極小（`rank × dim × 2 + max_loops × rank`），但讓每個迴圈的有效變換略微不同。

### 用法：加在 TransformerBlock 輸出上

`RecurrentBlock.forward`（`main.py:861-862`）：

```python
trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key=f"recurrent_loop_{t}")
trans_out = trans_out + self.lora(trans_out, t)   # LoRA delta 加在區塊輸出
```

測試 `test_different_loops_differ`（`tests/test_main.py:417-421`）確認 `loop_t=0` 與 `loop_t=1` 的 delta 不同。

## 4. 深度外推的關鍵設計：clamp

`AGENTS.md` 把這列為必須保留的不變量。`LoRAAdapter.forward`（`main.py:612-617`）：

```python
max_t = self.scale.num_embeddings - 1
t_idx = loop_t if loop_t <= max_t else max_t   # 超出訓練範圍時夾到最後一個
s = self.scale(torch.tensor(t_idx, device=x.device))
```

含義：**推理時 `n_loops` 可以超過訓練的 `max_loop_iters`**（深度外推，`README.md:246-250`）。
此時 Embedding 表沒有那麼多列，直接索引會越界。clamp 讓超出範圍的迭代**重用最後一個學到的尺度**，
而非崩潰。這是「深度外推能用」的工程前提。

> 📌 這也與 ACT 的「提早停止」互補：訓練範圍內靠 ACT 決定何時停，訓練範圍外靠 clamp 不讓 LoRA 越界。
> 兩者共同支撐「同一組權重跑更多次＝更深推理」的核心主張。

## 5. 兩個機制的分工

| 面向 | `loop_index_embedding` | `LoRAAdapter` |
|------|------------------------|---------------|
| 有無參數 | 無（純正弦） | 有（低秩 + 每迴圈尺度） |
| 注入位置 | 隱藏狀態 `h`（進 block 前） | 區塊輸出 `trans_out`（出 block 後） |
| 作用層次 | 提供「我在第幾圈」的訊號 | 對輸出做深度相關的微調 |
| 外推處理 | 函數定義域任意（無越界問題） | clamp 到最後一個尺度 |

兩者疊加：進 block 前知道深度（索引嵌入），出 block 後再按深度微調（LoRA）。
這讓「同一組權重、不同行為」在輸入側與輸出側都被照顧到。

## 6. 與第 04、05 篇的呼應

回想第 04 篇的迴圈五步管線，本文涵蓋其中 (a) 與 (d)：

```
(a) h_loop = loop_index_embedding(h, t, loop_dim)    ← 本文機制一
(b) combined = norm(h_loop + e)                      ← 第 04 篇（凍結 e）
(c) trans_out = TransformerBlock(combined, ...)      ← 第 06、07 篇
(d) trans_out += lora(trans_out, t)                  ← 本文機制二
(e) h = injection(h, e, trans_out)                   ← 第 05 篇（LTI）
(f) p = act(h)                                       ← 第 09 篇
```

至此五步只剩 (f) ACT 未談——那是下一篇的主題。

## 7. 本文結論與下篇預告

- 共享權重要能「按深度分化」，否則被迫做萬能折衷。
- `loop_index_embedding`：無參數正弦訊號，打在前 `dim//8` 通道，告訴 block「現在在第幾圈」。
- `LoRAAdapter`：共享低秩投影 + 每迴圈尺度向量，對區塊輸出做深度微調；clamp 支援深度外推。
- 兩者一前一後、一無參數一有參數，共同讓「迴圈不是單純重複」。

**下一篇（09）** 談最後一步 (f)：ACT（Adaptive Computation Time）如何讓每個位置**獨立決定**何時停止迴圈，
實現「簡單 token 早停、困難 token 多想」的自適應計算。

## 引用證據

- 問題動機：`README.md:316-321`
- `loop_index_embedding` 實作：`main.py:541-570`
- 迴圈內呼叫與 `loop_dim`：`main.py:858`，`main.py:821-823`
- `LoRAAdapter` 實作：`main.py:578-619`
- LoRA 在迴圈的用法：`main.py:861-862`
- clamp 深度外推不變量：`main.py:612-617`，`AGENTS.md`
- 測試：`tests/test_main.py:389-400`（索引嵌入），`tests/test_main.py:417-421`（LoRA）
- 來源論文：Relaxed Recursive Transformers（Bae et al., 2024），`README.md:353-363`
