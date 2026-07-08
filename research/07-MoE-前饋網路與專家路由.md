# 07 — MoE 前饋網路與專家路由

> 承接第 06 篇。上一篇談注意力（空間聚合）；本文談前饋（逐位置的非線性變換），
> 特別是 RecurrentBlock 內獨有的 **DeepSeekMoE 細粒度 MoE**。
> 這是「廣度」（跨領域能力）的來源，與迴圈提供的「深度」互補（`README.md:335-339`）。

## 1. 兩種 FFN，依位置選用

`TransformerBlock.__init__`（`main.py:650`）依 `use_moe` 旗標選 FFN：

```python
self.ffn = MoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3)
```

| 使用位置 | FFN | 類型 |
|----------|-----|------|
| RecurrentBlock 內的 TransformerBlock | `MoEFFN` | 稀疏路由 + 共享專家 |
| Prelude / Coda 的 TransformerBlock | `Expert`（稠密 SwiGLU） | 稠密 |

只有 `RecurrentBlock.__init__` 以 `use_moe=True` 建區塊（`main.py:816`）；
Prelude/Coda 都用預設 `use_moe=False`（`main.py:947, 951`）。

## 2. 單一專家：SwiGLU

`Expert`（`main.py:426-453`）是基本積木：

```python
def forward(self, x):
    return self.down(F.silu(self.gate(x)) * self.up(x))
```

即 gated linear unit 變體：`down(silu(gate(x)) * up(x))`。
它同時用作 MoE 內的個別路由專家，以及 Prelude/Coda 的稠密 FFN（後者 `expert_dim = dim * 4 // 3`）。

## 3. MoEFFN：兩類專家

`MoEFFN`（`main.py:456-533`）實作 DeepSeekMoE（Dai et al., 2024）。兩類專家（`main.py:460-470`）：

### 路由專家（routed experts）

- `n_experts` 個小 FFN（`main.py:487-489`）。
- 每個 token 由 router 選 top-`n_experts_per_tok` 個（`main.py:514`）。
- router 是 `nn.Linear(dim, n_experts)`（`main.py:483`）。

### 共享專家（shared experts）

- `n_shared_experts` 個較大的 FFN，**恆啟動**（`main.py:490-495`，`530-531`）。
- 寬度是 `expert_dim × n_experts_per_tok`（`main.py:492`）——刻意放大，吸收跨領域共用模式（語法、基本推理），
  這些若交給路由專家會被許多專家重複學到（`main.py:465-467`）。

### 啟動比例

每 token 啟動 `n_experts_per_tok / n_experts` 的路由容量 + 全部共享容量。
預設是 `4/64 = 6.25%`（`docs/open_mythos.md:81`）。

## 4. 無輔助損失的負載平衡（DeepSeek-V3 技巧）

這是 `MoEFFN.forward`（`main.py:508-516`）最精巧的部分。先看程式碼：

```python
logits = self.router(flat)                              # (B*T, n_experts)，無偏
scores = F.softmax(logits, dim=-1)
_, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)   # 選擇時加 bias
topk_scores = scores.gather(-1, topk_idx)               # 但權重用無偏 scores
topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)   # 重正規化
```

關鍵在於 **bias 只影響「選誰」，不影響「權重多大」**：

- `router_bias` 是 buffer 不是參數（`main.py:485`，測試 `test_router_bias_not_grad` `tests/test_main.py:363-366`），
  訓練時外部更新（用來把使用不足的專家更常選到）。
- 選 top-K 時用 `logits + router_bias`（`main.py:514`），但 gating 權重取自無偏的 `scores`（`main.py:515`）。
- 因此 bias **永遠不出現在梯度裡**——它只調整「派工」，不扭曲 loss 信號（`main.py:508-512` 註解）。

這避免了傳統 auxiliary loss 法會干擾主損失的問題。

## 5. 派工是 token 級 scatter

`main.py:518-527` 的迴圈把被選中的 token 送進對應專家：

```python
out = torch.zeros_like(flat)
for i in range(self.topk):
    expert_ids = topk_idx[:, i]
    token_scores = topk_scores[:, i].unsqueeze(-1)
    for eid in range(self.n_experts):
        mask = expert_ids == eid
        if not mask.any():
            continue
        out[mask] += token_scores[mask] * self.routed_experts[eid](flat[mask])
```

這是直觀但非最快的實作（逐專家 mask + 索引賦值）。研究用足夠；量產會換成 grouped GEMM。
重點是**正確性**：每個被選中的專家只處理分到它的 token，輸出按 gating 權重加總。

## 6. 共享專家恆啟動的測試保證

`test_shared_experts_always_fire`（`tests/test_main.py:368-375`）用一個巧妙方式驗證：
把**所有路由專家的權重清零**，若輸出仍非零，表示共享專家具獨立貢獻：

```python
for exp in self.moe.routed_experts:
    for p in exp.parameters():
        p.data.zero_()
out = self.moe(x)
assert out.abs().sum() > 0   # 共享專家仍貢獻
```

這保證了「共享專家不可能被誤設成路由的一部分」。

## 7. MoE 與迴圈的協同：每個深度可路由到不同專家

`README.md:337-339` 指出一個重要後果：隱藏狀態 `h_t` 跨迴圈演化，**router 在每個深度可能選到不同專家子集**，
於是「每個迴圈計算上都是獨特的，即使共享權重」。這讓 MoE 的「廣度」與迴圈的「深度」產生乘法效應：

- 深度：同一組權重反覆運算（第 04、05 篇）。
- 廣度：每步從大專家池選小子集（本文）。
- 兩者疊加：T 次迴圈 × 每步不同專家組合 ≈ 極大的計算路徑多樣性，但參數量只來自一組共享層 + 專家池。

這也是 `mythos_100b` 以上能把 `n_experts` 拉到 256–512、卻維持 `n_experts_per_tok=8`
（啟動比 ~1.5%）的原因——總參數大、每 token 計算小（第 03 篇變體表）。

## 8. 本文結論與下篇預告

- FFN 分兩種：稠密 `Expert`（Prelude/Coda）與 `MoEFFN`（RecurrentBlock）。
- MoE = 細粒度路由專家（稀疏）+ 共享專家（恆啟動），前者給廣度、後者吸收共用知識。
- 負載平衡靠「bias 只調選擇、不進梯度」的 DeepSeek-V3 技巧，無輔助損失干擾。
- MoE 廣度 × 迴圈深度 = 乘法式計算多樣性，是 RDT 參數效率的關鍵支柱。

**下一篇（08）** 回到迴圈本身：`loop_index_embedding` 與 `LoRAAdapter` 這兩個機制如何讓
「同一組權重」在每個迴圈迭代做出**功能上不同**的事——這是「迴圈不是單純重複」的來源。

## 引用證據

- FFN 選用：`main.py:650`，`main.py:816, 947, 951`
- `Expert` SwiGLU：`main.py:426-453`
- `MoEFFN` 兩類專家：`main.py:456-495`
- 無輔助損失平衡：`main.py:508-516`，`router_bias` 為 buffer `main.py:485`
- token 級 scatter 派工：`main.py:518-527`
- 共享專家測試：`tests/test_main.py:368-375`
- router_bias 非梯度測試：`tests/test_main.py:363-366`
- MoE×迴圈協同：`README.md:337-339`
- 啟動比例：`docs/open_mythos.md:81`
