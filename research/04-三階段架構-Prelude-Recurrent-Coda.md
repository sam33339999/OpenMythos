# 04 — 三階段架構：Prelude → Recurrent Block → Coda

> 承接第 03 篇。上一篇建立 `MythosConfig` 的參數語彙；本文沿著資料流走一遍
> 完整前向傳遞，並鎖定整個架構最重要的不變量：**凍結的編碼輸入 `e` 在每個迴圈被重新注入**。
> 這個不變量是後續（第 17 篇）把它對應到 Fusion「整合」的樞紐。

## 1. 整體資料流

`OpenMythos.forward`（`main.py:992-1034`）是唯一的前向入口。逐行對照：

```python
# main.py:1016-1034
T = input_ids.shape[1]
x = self.embed(input_ids)                       # 1. 嵌入
freqs_cis = (... mla 或 gqa ...)[start_pos:start_pos+T]  # 2. 選 RoPE
mask = self._causal_mask(T, ...) if T > 1 else None     # 3. 因果遮罩

for i, layer in enumerate(self.prelude):        # 4. Prelude
    x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")

e = x                                           # 5. 凍結編碼輸入 ← 核心不變量
x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)  # 6. 迴圈

for i, layer in enumerate(self.coda):           # 7. Coda
    x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")

return self.head(self.norm(x))                  # 8. 投影回詞表
```

三個階段在 `__init__` 中被組裝（`main.py:946-952`）：

- `prelude`：`nn.ModuleList`，每個是 `TransformerBlock(use_moe=False)`（稠密 FFN）
- `recurrent`：**單一個** `RecurrentBlock`（內含 MoE FFN）
- `coda`：`nn.ModuleList`，每個是 `TransformerBlock(use_moe=False)`

注意 Prelude 與 Coda **不用 MoE**，只有 Recurrent Block 內的 TransformerBlock 用 MoE（`main.py:650`，第 07 篇詳述）。

## 2. Prelude：把原始 token 變成「可注入的表示」

Prelude 的職責是把嵌入後的 token 序列「預處理」成一個穩定、資訊充分的表示 `e`。
它跑完 `prelude_layers` 個標準 transformer 區塊後，**這個輸出就被視為凍結的注入來源**（`main.py:1028`，`e = x`）。

為什麼要分出 Prelude 而不是直接把嵌入丟進迴圈？邏輯理由有二：

1. **注入訊號要「乾淨」**：`e` 在每個迴圈都以 `B·e` 的形式加回隱藏狀態（`main.py:742`）。
   如果 `e` 本身是未經注意力處理的原始嵌入，它缺乏上下文，反覆注入只會把模型拉回「字面」。
   先讓 Prelude 用注意力融上下文，`e` 才攜帶「這整段輸入在說什麼」的語意。
2. **關注點分離**：Prelude/Coda 是「一次性、可變層數」的標準計算；迴圈區塊是「共享權重、固定結構、可變深度」。
   分開後，加減 Prelude/Coda 層數不會動到迴圈的穩定性參數（第 05 篇）。

## 3. Recurrent Block：架構的心臟

`RecurrentBlock.forward`（`main.py:825-891`）跑最多 `n_loops` 次迭代，每次執行以下五步管線：

```python
# 對應 main.py:857-889
for t in range(n_loops):
    h_loop = loop_index_embedding(h, t, self.loop_dim)            # (a) 迴圈索引嵌入
    combined = self.norm(h_loop + e)                              # (b) 加上凍結 e 再正規化
    trans_out = self.block(combined, freqs_cis, mask,             # (c) TransformerBlock（注意力+MoE）
                           kv_cache, cache_key=f"recurrent_loop_{t}")
    trans_out = trans_out + self.lora(trans_out, t)               # (d) 深度 LoRA 微調
    h = self.injection(h, e, trans_out)                           # (e) LTI 穩定更新 h = A·h + B·e + trans_out

    p = self.act(h)                                               # (f) ACT 停止機率
    ... 累積權重、提早結束 ...
```

每個子步對應一個後續專篇：

| 步驟 | 機制 | 專篇 |
|------|------|------|
| (a) | `loop_index_embedding`（`main.py:541-570`） | 第 08 篇 |
| (b) | 凍結 `e` 注入 | 本文 + 第 17 篇 |
| (c) | `TransformerBlock`（`main.py:627-676`） | 第 06、07 篇 |
| (d) | `LoRAAdapter`（`main.py:578-619`） | 第 08 篇 |
| (e) | `LTIInjection`（`main.py:684-742`） | 第 05 篇 |
| (f) | `ACTHalting`（`main.py:750-780`） | 第 09 篇 |

### (b) 的關鍵：`combined = self.norm(h_loop + e)`

注意這一行（`main.py:859`）：注入發生在 **TransformerBlock 的正規化之前**，
也就是 `e` 是以「殘差相加」的形式進入注意力，而非作為額外的 conditioning。
這確保了 `e` 的維度必須與 `h` 相同（都是 `dim`），且注入是**加性、無參數**的（參數在 `(e)` 的 `B` 上）。

## 4. Coda：把迴圈後的潛在狀態「收尾」

Coda 接在迴圈後，跑 `coda_layers` 個標準區塊，把多次迴圈累積出的隱藏狀態映射回適合預測的表示，
最後過 `RMSNorm` 與權重共享的 LM head（`main.py:954-956`，`head.weight = embed.weight`）。

Coda 之所以不用 MoE，是因為「路由廣度」的價值在**反覆推理**的迴圈內（每個深度可路由到不同專家子集，
`README.md:337-339`）；Coda 只跑一次，MoE 的稀疏路由好處有限，稠密 FFN 更直接。

## 5. 權重共享與快取鍵命名

兩個維度的「共享」在這個架構裡很重要：

1. **權重共享**：LM head 與嵌入表共享權重（`main.py:956`），`test_weight_tying`（`tests/test_main.py:569-570`）驗證。
2. **迴圈權重共享**：整個 RecurrentBlock 是**一個** `TransformerBlock` 實例跑 `n_loops` 次（`main.py:816`）。
   這是「k 層參數、kL 層品質」的來源（`README.md:259-265`）。

KV 快取用字串鍵區分各層，避免碰撞：`"prelude_0"`、`"recurrent_loop_3"`、`"coda_1"` 等（`main.py:1026, 1032`，
`docs/open_mythos.md:247`）。這套命名對自迴歸生成至關重要（第 10 篇）。

## 6. 一個必須記住的細節：有快取時不提早結束

`RecurrentBlock.forward` 的提早結束有條件（`main.py:885-889`）：

```python
if halted.all() and kv_cache is None:
    break
```

也就是說：**只有在不使用 KV 快取時**，才會在所有位置都停止時提早跳出迴圈。
一旦有 KV 快取（生成模式），每個迴圈深度都必須跑完，這樣後續解碼步才能在每個 `cache_key` 找到已填入的鍵。
這是一個容易被忽略的正確性不變量——若誤改會破壞生成的快取一致性。

## 7. 本文結論與下篇預告

- 三階段是「一次性預處理 → 共享權重反覆 → 一次性收尾」的分工。
- **凍結的 `e` 在每個迴圈以 `B·e` 加回隱藏狀態**，是防止漂移的核心，也是與 Fusion「整合」對應的錨點。
- 快取鍵命名與「有快取不提早結束」是兩個隱性不變量。

**下一篇（05）** 深入 `(e)` 那一步——`LTIInjection` 如何用 ZOH 離散化**構造性地**保證譜半徑 ρ(A)<1，
讓迴圈訓練在任何學習率下都不發散。這是整個架構能訓練起來的數學根基。

## 引用證據

- `forward` 完整資料流：`main.py:992-1034`
- 三階段組裝：`main.py:946-952`
- 凍結 `e`：`main.py:1028`
- RecurrentBlock 五步管線：`main.py:825-891`（注入在 `859`，更新在 `863`）
- LTI 更新公式：`main.py:742`
- 權重共享：`main.py:956`，`tests/test_main.py:569-570`
- 有快取不提早結束：`main.py:885-889`
- 快取鍵命名：`main.py:1026, 1032`，`docs/open_mythos.md:247`
- Coda 不用 MoE 的理由：`README.md:337-339`
