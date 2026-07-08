# 09 — ACT 自適應計算時間

> 承接第 08 篇。至此迴圈五步管線只剩 (f) 未談。本文解析 `ACTHalting`——
> 讓**每個位置獨立**決定何時停止累積更新的機制。這是「簡單 token 早停、困難 token 多想」的來源，
> 也是 Universal Transformer 賦予模型 Turing 完備性的關鍵（`README.md:326-333`）。

## 1. 動機：過度思考問題

「更多迴圈」並非總是更好。`README.md:326-331` 指出失敗模式：

> 超過某個深度後，過度遞迴會**降低**預測——隱藏狀態漂移過解答、進入雜訊。這是「overthinking」。

若每個 token 都無腦跑滿 `max_loop_iters`，簡單 token 會被過度處理、困難 token 可能還沒想完就被截斷。
需要的不是「固定深度」，而是「**學會何時收斂**」。

## 2. `ACTHalting`：逐位置停止機率

`ACTHalting`（`main.py:750-780`）極簡——一個線性層加 sigmoid：

```python
class ACTHalting(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.halt = nn.Linear(dim, 1)
    def forward(self, h):
        return torch.sigmoid(self.halt(h)).squeeze(-1)   # (B, T) ∈ (0,1)
```

它在每個迴圈步從當前隱藏狀態預測一個「本步的停止機率」`p`（per position）。
測試 `test_values_in_01`（`tests/test_main.py:506-510`）驗證輸出落在 [0,1]，`test_output_shape`（`502-504`）驗證形狀 `(B, T)`。

## 3. ACT 剩餘機率技巧（remainder trick）

真正的學問在 `RecurrentBlock.forward` 的累積邏輯（`main.py:865-883`）。逐段解析：

```python
# main.py:853-855
halted = torch.zeros(B, T, dtype=torch.bool)        # 已停止的位置
cumulative_p = torch.zeros(B, T)                    # 累積停止機率
h_out = torch.zeros_like(h)                         # ACT 加權輸出
```

每個迴圈步：

```python
# main.py:865-880
p = self.act(h)                          # 本步停止機率 (B,T)
still_running = ~halted                  # 還沒停的位置

remainder = (1.0 - cumulative_p).clamp(min=0)    # 剩餘機率質量
weight = torch.where(
    cumulative_p + p >= self.cfg.act_threshold,  # 若累積+本步會跨過門檻…
    remainder,                                   # …就把剩餘全給它（最後一步）
    p,                                           # 否則給本步機率
)
weight = weight * still_running.float()          # 已停的位置權重歸零
h_out = h_out + weight.unsqueeze(-1) * h         # 加權累加隱藏狀態

cumulative_p = cumulative_p + p * still_running.float()
halted = halted | (cumulative_p >= self.cfg.act_threshold)
```

這是 Graves (2016) ACT 的標準「remainder trick」，但有一個 OpenMythos 特有的修正（見下節）。

### 為什麼要 remainder trick

直觀：我們希望輸出是「跨迴圈隱藏狀態的加權和」，權重和為 1。但若累積停止機率在倒數第二步
剛好是 0.97、門檻 0.99，本步 p=0.05，相加 1.02 > 0.99 觸發停止——此時若直接用 p=0.05 會超過 1。
remainder 把「跨過門檻那步」的權重設成 `1 - cumulative_p`，確保權重總和恰為 1（機率質量守恆）。

## 4. OpenMythos 的修正：`still_running` 閘門

注意 `weight = weight * still_running.float()`（`main.py:879`）。`main.py:868-872` 的註解解釋為何必要：

> 一旦某位置跨過門檻，它在停止那步貢獻一次（remainder），之後必須**零貢獻**。
> 否則當 `act_threshold < 1` 時，`remainder` 在後續每步都非零，會每步洩漏。

換句話說：若不乘上 `still_running`，已停止位置的 `remainder`（= `1 - cumulative_p`）在停止後仍 > 0，
會被重複加進 `h_out`，破壞「只貢獻一次」的語意。這個閘門確保每個位置**恰好在停止那步**貢獻完剩餘質量，之後靜默。

## 5. 提早結束的條件（與第 04 篇呼應）

```python
# main.py:888-889
if halted.all() and kv_cache is None:
    break
```

第 04 篇已強調：**只有無 KV 快取時才提早跳出**。有快取時即使全停也要跑完所有迴圈，
以保證每個 `cache_key` 都被填入（否則後續解碼找不到鍵）。這是 ACT 與 KV 快取正確互動的硬性約束。

## 6. ACT 帶來的三個性質

`README.md:326-333` 與 `docs/open_mythos.md:316-318` 綜合：

1. **逐位置自適應計算**：同一個 batch 內，簡單 token 早停、困難 token 跑滿深度。
   這是「Continuous Depth-wise Batching」的基礎（`README.md:366-371`），理論上可帶來 2–3× 吞吐提升。
2. **防過度思考**：收斂後的位置停止累積更新，避免漂進雜訊區。
3. **理論上的 Turing 完備性**：Graves (2016) 證明，在對 transformer block 表達力的某些假設下，
   ACT 讓模型 Turing 完備——這對「能解決的問題類別」有理論意涵（`README.md:331-332`）。

## 7. 門檻的取捨

`act_threshold` 預設 0.99（`main.py:73`），所有變體也都是 0.99（`variants.py`）。
含義：

- 越接近 1：位置傾向跑更多步才停（更多計算、更深推理），但更易觸及過度思考。
- 越低：提早停止更激進，省計算但可能沒想透。

0.99 是個偏「讓它多想」的設定，與 RDT「靠深度取勝」的哲學一致。但它也意味著多數位置會跑到很深——
這正是第 19 篇會討論的「過度思考」風險來源。

## 8. 本文結論與下篇預告

- ACT 讓每個位置學一個停止機率，跨迴圈累積；remainder trick 保證權重和為 1。
- `still_running` 閘門修正「已停位置重複洩漏」的問題，是 OpenMythos 的關鍵細節。
- 提早結束僅在無 KV 快取時啟用；有快取時必須跑滿以維持快取一致性。
- 帶來逐位置自適應計算、防過度思考、理論 Turing 完備性。
- 門檻 0.99 偏「多想」，與 RDT 深度哲學一致，但也埋下過度思考風險。

**下一篇（10）** 把前幾篇的「單次前向」串成「自迴歸生成」：KV 快取如何運作、`generate()` 的解碼迴圈、
以及「推理時加迴圈數」的深度外推如何兌現成更難問題的解答。

## 引用證據

- 過度思考問題：`README.md:326-333`
- `ACTHalting` 實作：`main.py:750-780`
- remainder trick 與 still_running 閘門：`main.py:853-883`（註解 `868-872`）
- 提早結束條件：`main.py:888-889`
- 門檻預設：`main.py:73`，`variants.py`
- ACT 測試：`tests/test_main.py:502-510`
- Turing 完備與連續深度批次：`README.md:331-332`，`README.md:366-371`，`docs/open_mythos.md:316-318`
- 來源：Graves (2016)，`docs/open_mythos.md:317`
