# 05 — LTI 穩定注入與譜半徑

> 承接第 04 篇。上一篇指出 RecurrentBlock 的第 (e) 步 `h = A·h + B·e + trans_out` 是穩定性的樞紐；
> 本文拆解 `LTIInjection` 如何用**動態系統理論**讓這個遞迴「構造性地」永不發散。
> 這是整個 OpenMythos 能被訓練起來的數學根基，也是本系列後續討論「回授」時最核心的理論章節。

## 1. 為什麼迴圈訓練會不穩定

README「The Stability Problem」(`README.md:268-273`) 列出兩個主要失敗模式：

- **殘差爆炸**：隱藏狀態 `h_t` 隨迴圈次數無界增長。
- **Loss 尖峰**：注入參數的譜範數過大，訓練突然發散。

直觀原因：遞迴 `h_{t+1} = A·h_t + …` 把 `A` 反覆相乘。若 `A` 的特徵值有任何一個絕對值 ≥ 1，
對應分量就會以該特徵值為底呈指數增長，T 次後爆炸。

## 2. 動態系統視角：譜半徑決定一切

把非線性的 Transformer 貢獻先忽略，遞迴退化為離散線性非時變（LTI）系統（`README.md:276-283`）：

```
h_{t+1} = A·h_t + B·e
```

對 LTI 系統，穩定性**完全**由 `A` 的譜半徑 ρ(A)（= 最大特徵值絕對值）決定：

| 條件 | 行為 |
|------|------|
| ρ(A) < 1 | 穩定、收斂（齊次解衰減到零） |
| ρ(A) ≥ 1 | 不穩定、發散 |

README 給出經驗觀察（`README.md:287-288`）：**每一次發散的訓練，學到的 ρ(A) ≥ 1；
每一次收斂的訓練，都維持 ρ(A) < 1**。這不是巧合，而是上述定理的直接體現。

## 3. OpenMythos 的解法：構造性保證

`LTIInjection`（`main.py:684-742`）不靠正則化或梯度懲罰來「鼓勵」穩定，
而是用**參數化**讓 ρ(A)<1 **在數學上不可能被違反**。做法分三步（`README.md:291-298`，`main.py:696-698`）：

### 步驟一：把 A 參數化為連續負對角矩陣

```
A_continuous = Diag(-exp(log_A))     # 永遠是負對角
```

`log_A` 是可學參數（`main.py:710`，初始化為 `torch.zeros(dim)`）。
`exp(log_A)` 恆正，前置負號後對角元素恆負。

### 步驟二：ZOH（零階保持）離散化

```
A_discrete = exp(Δt · A_continuous)  # 逐元素，值落在 (0, 1)
```

其中 `Δt = exp(log_dt)`，`log_dt` 也是可學參數（`main.py:711`，初始化為 `torch.zeros(1)`）。
因為 `A_continuous` 的對角元素是負的，`Δt > 0`，所以 `Δt · A_continuous` 的對角元素也是負的，
而「負數的指數」`exp(負)` 必落在 `(0, 1)`。

### 步驟三：在 log 空間計算以避免數值 NaN

`get_A()` 的實作（`main.py:714-725`）：

```python
def get_A(self) -> torch.Tensor:
    return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))
```

為什麼先合併 `log_dt + log_A` 再取兩次 exp？因為（`main.py:722-725` 註解）：

```
dt · A_c = -exp(log_dt) · exp(log_A) = -exp(log_dt + log_A)
```

在 log 空間相加可避免「0 × inf = NaN」的退化（當 `log_dt → -∞`、`log_A → +∞` 時）。
`.clamp(-20, 20)` 把積限制在 float32 可表示範圍，保證任何梯度步長下都有限。

### 結果

`A_discrete` 的每個對角元素嚴格落在 `(0, 1)`，所以 ρ(A) = max(對角元素) < 1 **必然成立**，
與學習率、batch 雜訊無關。`README.md:296-298` 稱此為 Parcae 架構（Prairie et al., 2026）。

## 4. 更新規則的完整形式

`forward`（`main.py:727-742`）：

```python
def forward(self, h, e, transformer_out):
    A = self.get_A()
    return A * h + self.B * e + transformer_out
```

- `A * h`：衰減項。每一個通道獨立衰減（對角），衰減率由學習決定但受限於 (0,1)。
- `self.B * e`：注入項。`B` 是可學參數（`main.py:712`，初始化 `0.1`），把凍結的 `e` 加回。
- `transformer_out`：非線性推理貢獻（第 04 篇的 attention + MoE 輸出）。

三者相加：**衰減的舊狀態 + 重新注入的原訊號 + 本輪新推理**。這正是第 04 篇強調的「不漂移」機制的具體實現。

## 5. 測試如何驗證「構造性保證」

`tests/test_main.py` 有四個針對性測試：

| 測試 | 斷言 | 行號 |
|------|------|------|
| `test_output_shape` | 輸出形狀正確 | `tests/test_main.py:465-469` |
| `test_spectral_radius_lt_1` | `A.max() < 1.0` | `tests/test_main.py:471-473` |
| `test_spectral_radius_gt_0` | `A.min() > 0.0`（嚴格大於 0） | `tests/test_main.py:475-477` |
| `test_spectral_radius_stable_after_large_grad_step` | **lr=1e3** 的 SGD 一步後 `A.max() < 1` 仍成立 | `tests/test_main.py:479-489` |

最後一個測試最有說服力：它模擬一個「激進到荒謬」的梯度更新（lr=1000），
然後斷言穩定性**仍然**成立。這正是「構造性保證」的含義——不依賴溫和的訓練設定。

> 📌 注意這些測試用 `A.max()` 而非 `eigvals`，因為 `get_A()` 回傳的是 1-D 對角向量（`main.py:719`），
> 其 `max()` 即等於譜半徑。但若要套用到一般（非對角）矩陣，必須用 `torch.linalg.eigvals(A).abs().max()`，
> 如 `README.md:99-103` 所示。`example.py:49` 用 `A.max()` 被標為「錯誤」是因為它誤稱之為「譜半徑」，
> 但在當前對角實作下數值恰好正確。

## 6. 從穩定性到「回授」的理論橋梁

這裡埋下本系列第 16 篇的種子。LTI 注入告訴我們：**一個穩定的回授系統，其回授矩陣的譜半徑必須 < 1**。
這是控制理論的基本結論。換句話說：

- 「迴圈深度推理」= 一個離散動態系統。
- 「能收斂」= 系統是漸近穩定的。
- 「漸近穩定」⇔ ρ(A) < 1。

OpenMythos 的貢獻不是發現這個條件（控制理論早有），而是**把它編進參數化**，
讓梯度下降無論怎麼走都踩在穩定區內。這個思路對「設計自己的回授 LLM」極具啟發——
後續設計文件（第 21 篇）會把它列為第一條硬性約束。

## 7. 本文結論與下篇預告

- 穩定性由 ρ(A)<1 決定，這是動態系統定理，非經驗觀察。
- OpenMythos 用「連續負對角 + ZOH 離散化 + log 空間計算」**構造性**保證 ρ(A)∈(0,1)。
- `B·e` 注入與 `A·h` 衰減共同構成「不忘原訊號、又容許演化」的回授結構。
- 激進梯度步驟後仍穩定，有測試直接驗證。

**下一篇（06）** 轉向另一個支柱——注意力機制：MLA 與 GQA 如何在「表達力」與「KV 快取成本」之間取捨，
以及為什麼所有正式變體都選了 MLA。

## 引用證據

- 穩定性問題與譜半徑條件：`README.md:268-298`
- `LTIInjection` 全實作：`main.py:684-742`
- `get_A` 的 log 空間計算與 clamp：`main.py:714-725`
- 更新規則三項：`main.py:742`
- 穩定性測試（含 lr=1e3）：`tests/test_main.py:471-489`
- 譜半徑正確計算方式：`README.md:99-103`，`AGENTS.md`
