# 03 — MythosConfig 與模型變體

> 承接第 02 篇。上一篇講完「怎麼跑」；本文解析模型唯一的設定入口 `MythosConfig`，
> 以及 1B–1T 七個變體如何用「同一份 dataclass」表達迥異的尺度。
> 這些欄位是後續各篇（注意力、MoE、LTI、ACT）的共同詞彙表。

## 1. 唯一的設定入口：`MythosConfig`

`MythosConfig` 是一個 `@dataclass`（`main.py:16-81`），所有超參數都集中在此。
`OpenMythos.__init__` 只接收一個 `cfg: MythosConfig`（`main.py:926`），
而每個子模組也都從同一份 `cfg` 讀自己需要的欄位（例如 `MLAttention.__init__` 讀 `cfg.kv_lora_rank`，`main.py:320`）。

### 核心欄位（`main.py:51-58`）

| 欄位 | 預設 | 意義 |
|------|------|------|
| `vocab_size` | 32000 | 詞表大小；嵌入表與 LM head 的維度 |
| `dim` | 2048 | 模型隱藏維度（殘差流的寬度） |
| `n_heads` | 16 | query 注意力頭數 |
| `n_kv_heads` | 4 | key/value 頭數（**僅 GQA**；MLA 忽略） |
| `max_seq_len` | 4096 | RoPE 預計算的最大序列長度 |
| `max_loop_iters` | 16 | 推理時的預設迴圈深度 T |
| `prelude_layers` | 2 | 迴圈前的標準層數 |
| `coda_layers` | 2 | 迴圈後的標準層數 |

### 注意力欄位（`main.py:60-66`）

`attn_type` 決定走哪一套完整實作：

| 欄位 | 預設 | 意義 |
|------|------|------|
| `attn_type` | `"mla"` | `"gqa"` → GQAttention；`"mla"` → MLAttention（`main.py:649`） |
| `kv_lora_rank` | 512 | **[MLA]** 快取中儲存的壓縮 KV 潛在維度（而非完整 K/V） |
| `q_lora_rank` | 1536 | **[MLA]** 壓縮 Q 潛在維度 |
| `qk_rope_head_dim` | 64 | **[MLA]** 每頭接收 RoPE 的維度 |
| `qk_nope_head_dim` | 128 | **[MLA]** 每頭不帶位置編碼的維度 |
| `v_head_dim` | 128 | **[MLA]** 每頭 value 維度 |

MLA 的快取大小 = `kv_lora_rank + n_heads × qk_rope_head_dim`，
對照 GQA 快取 = `n_kv_heads × head_dim × 2`——量產規模下 MLA 約小 10–20 倍（`docs/open_mythos.md:68`，`main.py:284-309`）。

### MoE 前饋欄位（`main.py:68-71`）

MoE FFN **只**用在 Recurrent Block 內；Prelude/Coda 用稠密 SwiGLU（`main.py:650`，`docs/open_mythos.md:72`）。

| 欄位 | 預設 | 意義 |
|------|------|------|
| `n_experts` | 64 | 路由專家總數 |
| `n_shared_experts` | 2 | 恆啟動的共享專家數 |
| `n_experts_per_tok` | 4 | 每 token 由路由器選的 top-K |
| `expert_dim` | 512 | 每個細粒度專家的內部隱藏維度 |

每 token 約啟動 `n_experts_per_tok / n_experts = 6.25%` 的路由專家容量，加上全部共享專家容量（`docs/open_mythos.md:81`）。

### 穩定性與調適欄位（`main.py:73-81`）

| 欄位 | 預設 | 意義 |
|------|------|------|
| `act_threshold` | 0.99 | ACT 累積停止機率門檻 |
| `rope_theta` | 500000.0 | RoPE 基頻（LLaMA-3 預設） |
| `lora_rank` | 16 | 每個迴圈的深度 LoRA 適配器秩 |
| `max_output_tokens` | 4096 | 每次前向傳遞最大生成 token 數 |
| `dropout` | 0.0 | dropout（0 關閉；0.1 為預訓練標準） |

## 2. 七個預設變體：1B 到 1T

`open_mythos/variants.py` 提供 7 個工廠函式，各自回傳一個調好數值的 `MythosConfig`
（`variants.py:9-198`）。README 的對照表（`README.md:131-139`）：

| 變體 | `dim` | 專家數 | `expert_dim` | 迴圈數 | 上下文 | 最大輸出 |
|------|-------|--------|--------------|--------|--------|----------|
| `mythos_1b` | 2048 | 64 | 2048 | 16 | 4k | 4k |
| `mythos_3b` | 3072 | 64 | 4096 | 16 | 4k | 4k |
| `mythos_10b` | 4096 | 128 | 5632 | 24 | 8k | 4k |
| `mythos_50b` | 6144 | 256 | 9728 | 32 | 8k | 4k |
| `mythos_100b` | 8192 | 256 | 13568 | 32 | 1M | 128k |
| `mythos_500b` | 12288 | 512 | 23040 | 48 | 1M | 128k |
| `mythos_1t` | 16384 | 512 | 34560 | 64 | 1M | 128k |

### 可觀察的尺度法則

從小到大讀這張表，能歸納出 OpenMythos 預設的**擴展策略**（這也是第 20 篇訓練策略的依據）：

1. **`dim` 約略 ×√2 成長**：2048 → 16384，每兩階約翻倍。這是殘差流寬度，直接決定每層計算量。
2. **專家數在 50B 之後才大幅增加**（64 → 128 → 256 → 512）：代表「廣度」（領域覆蓋）被當作
   大模型的差異化能力，小模型先用較少專家。
3. **迴圈數單調上升**（16 → 64）：這是 RDT 最獨特的擴展軸——**模型越大、預設「想得越深」**。
   一般 Transformer 不存在這個維度。`README.md:248-250` 把這點連到「深度外推」。
4. **上下文只在 100B 跳到 1M**：長脈絡是最大規模才啟用的能力，伴隨 `rope_theta` 從 500k 升到 1M–2M（`variants.py:139, 167, 195`）。
5. **`n_shared_experts` 從 2 增到 8**：越大越強調「跨領域共用知識」的吸收（`main.py:456-470` 的設計意圖）。
6. **`n_experts_per_tok` 從 4 增到 8**：在大模型把每 token 啟動比例維持在 ~1.5%（8/512）的低稀疏度。

> 📌 一個關鍵觀察：**所有變體的 `attn_type` 都是 `"mla"`**（`variants.py` 逐筆可見）。
> 這與 README「Model Variants」範例誤稱 `mythos_7b()`（根本不存在）的 GQA 框架不一致——
> `AGENTS.md` 已明確標注此 doc/code 不一致。實際可用的是上表 7 個。

## 3. 變體的參數預算公式

`variants.py:3-6` 的註解給出預算拆解：

```
total ≈ embed + prelude/coda 稠密區塊 + recurrent MLA + MoE
MoE   = 3 * dim * expert_dim * (n_experts + n_shared * n_experts_per_tok)
```

這表示 `expert_dim` 是從「扣除其他項後的剩餘預算」反解出來的——所以它不是任意值，
而是讓總參數數對齊到目標量級（1B/3B/…）的應變數。這也是為何 `expert_dim` 隨 `dim` 一起長。

## 4. 使用變體的最小範例

```python
from open_mythos import mythos_3b, OpenMythos

cfg = mythos_3b()
cfg.vocab_size = 1000          # 實際訓練時會覆寫成 tokenizer 的 vocab_size
model = OpenMythos(cfg)
print(f"參數量: {sum(p.numel() for p in model.parameters()):,}")
```

訓練腳本就是這樣用的：`training/3b_fine_web_edu.py:397-399` 先 `cfg = mythos_3b()`，
再把 `cfg.vocab_size` 設成 tokenizer 的詞表大小、`cfg.max_seq_len` 設成 2048。

## 5. 本文結論與下篇預告

- `MythosConfig` 是**唯一**設定入口，所有子模組都從它讀欄位——設計上高度集中。
- 七個變體揭示 RDT 的擴展哲學：**隨規模增大，不只加寬加專家，還加迴圈深度**。
- MLA 是所有正式變體的預設注意力（不是 GQA），長脈絡能力在大規模才解鎖。

**下一篇（04）** 將進入真正的架構資料流：Prelude → Recurrent Block → Coda 的三階段接力，
特別是「在 Prelude 結束時凍結 `e` 並在每個迴圈重新注入」這個核心不變量。

## 引用證據

- `MythosConfig` 全欄位：`main.py:16-81`
- 注意力選擇邏輯：`main.py:649`
- MLA 快取大小對比：`main.py:284-309`，`docs/open_mythos.md:68`
- MoE 只在 Recurrent Block：`main.py:650`，`docs/open_mythos.md:72`
- 變體定義與預算公式：`variants.py:3-6, 9-198`
- 變體對照表：`README.md:131-139`
- README `mythos_7b` 不存在的不一致：`AGENTS.md`
- 訓練腳本使用變體：`training/3b_fine_web_edu.py:397-399`
