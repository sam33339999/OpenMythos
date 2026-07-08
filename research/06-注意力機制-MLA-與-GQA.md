# 06 — 注意力機制：MLA 與 GQA

> 承接第 05 篇。上一篇談穩定性（時間維度的回授）；本文談注意力（空間維度的資訊聚合）。
> OpenMythos 把注意力做成可替換的，靠 `cfg.attn_type` 在 MLA 與 GQA 之間切換（`main.py:649`）。
> 本文解析兩者的差異、KV 快取成本，以及「為何所有正式變體都用 MLA」。

## 1. 可替換的注意力：`TransformerBlock` 的分派

`TransformerBlock.__init__`（`main.py:640-651`）依 `cfg.attn_type` 決定實例化哪個類別：

```python
# main.py:649
self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
```

測試 `test_attn_type_selection`（`tests/test_main.py:451-453`）直接斷言這個分派：
`gqa` 配置 → `GQAttention` 實例；`mla` 配置 → `MLAttention` 實例。

兩者都遵守同一個 `forward` 簽名（`x, freqs_cis, mask, kv_cache, cache_key`），所以對上層完全透明。

## 2. GQA：Grouped Query Attention

`GQAttention`（`main.py:177-276`）實作 Ainslie et al., 2023。

### 核心想法

query 頭數 `n_heads` 多於 key/value 頭數 `n_kv_heads`（`main.py:201-204`）：
每個 KV 頭被 `n_heads // n_kv_heads` 個 query 頭共享，把 KV 快取大小縮成該比例分之一，但保留完整 query 表達力。

預設 `MythosConfig` 是 `n_heads=16, n_kv_heads=4`（`main.py:53-54`），即每 4 個 query 頭共用 1 對 KV——快取省 4 倍。

### RoPE 在快取**之前**套用

`forward`（`main.py:236-237`）先對 Q、K 做 RoPE，**再**寫入快取：

```python
q = apply_rope(q, freqs_cis)
k = apply_rope(k, freqs_cis)
# ...之後才寫 kv_cache (main.py:239-243)
```

這是 `AGENTS.md` 列為不變量的設計：**RoPE 在進入 KV 快取前套用，所以快取值已帶位置編碼，取出時不必重旋**。
若違反此順序，解碼時會對已旋轉的鍵再旋一次，破壞相對位置語意。

### Flash Attention 2 的選用與退回

`main.py:245-274` 是關鍵的不對稱：

```python
if _HAS_FLASH_ATTN:
    # flash_attn_func 原生支援 GQA（不需 repeat_interleave），轉 bf16
    out = flash_attn_func(q, k, v, dropout_p=..., causal=(mask is not None))
else:
    # 手動 SDPA：需把 KV 頭展開到與 Q 相同
    k = k.repeat_interleave(self.groups, dim=2)
    ...
```

`AGENTS.md` 特別強調：**Flash Attention 只影響 `GQAttention`，不影響 `MLAttention`**（MLA 走自己的 matmul 路徑）。
沒裝 flash-attn 時靜默退回，由 `_HAS_FLASH_ATTN` 守護。

## 3. MLA：Multi-Latent Attention（DeepSeek-V2 風格）

`MLAttention`（`main.py:284-418`）是所有正式變體的選擇。它的核心洞見（`main.py:286-298`）：

> 與其快取完整的 K 與 V（各 `n_heads × head_dim` 每 token），不如把 KV 路徑壓縮過一個低秩潛在 `c_kv`，
> 只快取這個潛在加上 RoPE 鍵；解碼時再用便宜的線性投影重建 K_nope 與 V。

### Q 路徑（壓縮 → 展開）

`main.py:371-376`：

```
x → q_down (dim→q_lora_rank) → q_norm
  → q_up_nope (q_lora_rank → n_heads×qk_nope_head_dim)  [無 RoPE]
  → q_up_rope (q_lora_rank → n_heads×qk_rope_head_dim)  [套 RoPE]
q = cat(q_nope, q_rope)  每頭
```

### KV 路徑（壓縮 → 快取 → 重建）

`main.py:378-404`：

```
x → kv_down (dim → kv_lora_rank + qk_rope_head_dim)
     分成 c_kv（潛在，快取）與 k_rope_raw（跨頭共享）
k_rope = RoPE(expand(k_rope_raw))   ← 快取前套 RoPE（與 GQA 同一不變量）
c_kv → kv_norm → kv_up → [k_nope | v]  每步重建
k = cat(k_nope, k_rope)  每頭
```

### 快取內容

`forward` 把 `c_kv`（`kv_lora_rank`）與 `k_rope`（`n_heads × qk_rope_head_dim`）寫入快取（`main.py:391-395`）。
測試 `test_cache_stores_compressed_kv`（`tests/test_main.py:308-315`）斷言 `c_kv` 的最末維等於 `kv_lora_rank`，而非完整 K/V。

快取大小對比（`docs/open_mythos.md:68`，`main.py:307-309`）：

| 注意力 | 每層每 token 快取 |
|--------|-------------------|
| GQA | `n_kv_heads × head_dim × 2` |
| MLA | `kv_lora_rank + n_heads × qk_rope_head_dim` |

量產規模下 MLA 約小 **10–20 倍**——這對 1M 上下文（`mythos_100b` 以上）是決定性的。

## 4. 為什麼正式變體全用 MLA

對照第 03 篇的變體表，七個變體的 `attn_type` 無一例外都是 `"mla"`（`variants.py` 逐筆）。
理由可從尺度推導：

1. **長脈絡只在 100B 以上解鎖**（`max_seq_len=1000000`）。在 1M token 下，KV 快取是顯存瓶頸，
   MLA 的 10–20× 縮減是「能不能塞得進去」的差別，而非「省不省錢」。
2. **MLA 的重建成本可接受**：`kv_up` 只是一個矩陣乘法（`main.py:400`），相較於快取省下的顯存頻寬，划算。
3. **GQA 的優勢在「短脈絡、低延遲」**：它不必重建，且能吃 Flash Attention 2 的最佳化。
   小研究模型或單測（`tests/test_main.py` 的 `gqa_cfg`，`main.py` 的 `example.py`）用 GQA 是因為簡單、快。

這解釋了為何 `example.py` 與單測偏好 GQA，但正式變體清一色 MLA——**尺度決定了 KV 快取是主導成本**。

## 5. MLA 快取比 GQA 小的實測

`test_mla_fewer_kv_cache_bytes`（`tests/test_main.py:659-674`）直接量化這點：

```python
# 同一個序列、同樣設定，分別跑 GQA 與 MLA，比較快取位元組數
def cache_bytes(cache):
    return sum(t.numel() * t.element_size() for entry in cache.values() for t in entry.values())
assert cache_bytes(cache_mla) < cache_bytes(cache_gqa)
```

這個測試用最小配置（`dim=64`）就成立，說明 MLA 的快取優勢在很小規模就顯現，而非只在量大時才有。

## 6. 兩者共用的不變量：RoPE 先於快取

無論 MLA 或 GQA，都遵守「RoPE 套用於寫入快取**之前**」：

- GQA：`main.py:236-243`
- MLA：`main.py:389-395`（`k_rope = apply_rope(...)` 後才寫 cache）

`AGENTS.md` 把這條列為架構不變量。理由已在第 2 節說明：避免取出時重旋。
設計新注意力時若偏離此序，會破壞增量解碼的正確性。

## 7. 本文結論與下篇預告

- 注意力是可替換零件，靠 `attn_type` 分派；對上層透明。
- GQA：query 多、KV 少，靠共享省快取；吃 Flash Attention 2（選用），否則手動 SDPA。
- MLA：壓縮 KV 潛在 + RoPE 鍵，快取小 10–20×，量產長脈絡的唯一實用選項。
- 兩者共用「RoPE 先於快取」不變量。
- 正式變體全用 MLA，因為尺度下 KV 快取是主導成本。

**下一篇（07）** 進入迴圈區塊的 FFN：DeepSeekMoE 的細粒度專家 + 共享專家 + 無輔助損失負載平衡，
看「廣度」是如何在不炸參數量的前提下被加進來的。

## 引用證據

- 注意力分派：`main.py:649`，測試 `tests/test_main.py:451-453`
- GQA 實作與 RoPE 順序：`main.py:177-276`（RoPE 在 `236-237`，快取在 `239-243`）
- Flash 退回：`main.py:245-274`，`AGENTS.md`
- MLA 實作：`main.py:284-418`（Q 路徑 `371-376`，KV 路徑 `378-404`，快取 `391-395`）
- 快取大小對比：`docs/open_mythos.md:68`，`main.py:307-309`
- MLA 快取壓縮測試：`tests/test_main.py:308-315`
- MLA 快取更小測試：`tests/test_main.py:659-674`
- 變體全用 MLA：`variants.py:9-198`，`AGENTS.md`
- RoPE 先於快取不變量：`AGENTS.md`
