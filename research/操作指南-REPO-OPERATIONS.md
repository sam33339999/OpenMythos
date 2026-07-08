# OpenMythos Repo 操作指南（繁體中文）

> 本指南基於**實際讀取的原始碼**撰寫——每個指令、檔案路徑、參數都對應到倉庫內的真實程式碼。
>
> **驗證狀態聲明**：撰寫本指南時，本機環境為 Python 3.14 且**未安裝 torch**，因此指令未在此環境實跑。
> 所有指令皆從原始碼的 `argparse` 定義、進入點、`__main__` 區塊直接萃取，在正確環境（Python 3.10–3.13 + torch==2.11.0）下可執行。
> 請在你的機器上依下方步驟安裝後驗證。

---

## 目錄

1. [環境建置](#1-環境建置)
2. [倉庫全景：每個檔案做什麼](#2-倉庫全景每個檔案做什麼)
3. [兩套模型：OpenMythos vs MoDA](#3-兩套模型openmythos-vs-moda)
4. [小型確認（Small-Scale Verification）](#4-小型確認small-scale-verification)
5. [完整訓練](#5-完整訓練)
6. [已知地雷與限制](#6-已知地雷與限制)

---

## 1. 環境建置

### 1.1 前置條件

| 項目 | 需求 | 出處 |
|------|------|------|
| Python | `>=3.10, <4.0`（實務建議 3.10–3.13，3.14 可能無對應 torch wheel） | `pyproject.toml:40` |
| torch | **`==2.11.0`（精確釘版）** | `pyproject.toml:41` |
| 建置系統 | Poetry（`poetry-core`） | `pyproject.toml:1-4` |

> ⚠️ **權威相依在 `pyproject.toml`，不是根目錄的 `requirements.txt`**。
> 根目錄 `requirements.txt` 是鬆散的手動清單（`torch>=2.1.0`，`requirements.txt:1`），
> 與 pyproject 的精確釘版（`torch==2.11.0`）**不一致**。一律以 pyproject 為準（`AGENTS.md` 已記載）。

### 1.2 安裝

```bash
# 方式 A：pip 可編輯安裝（開發推薦）
pip install -e .

# 方式 B：Poetry
poetry install

# （選用）Flash Attention 2 —— 只影響 GQAttention，需 CUDA + 建置工具
pip install open-mythos[flash]
```

### 1.3 驗證安裝成功

```bash
python -c "from open_mythos import OpenMythos, MythosConfig; print('import OK')"
```

> ⚠️ `from open_mythos import load_tokenizer` 會 **ImportError**——`__init__.py:52` 把它列進 `__all__` 但從未匯入（`AGENTS.md` 已記載）。別用這個名字。

### 1.4 程式碼品質工具

```bash
ruff check .     # lint（pyproject.toml:66-67，行長 88）
black .          # 格式化（驗證：black --check .，target py310）
```

---

## 2. 倉庫全景：每個檔案做什麼

```
OpenMythos/
├── open_mythos/           ← 套件本體（發布名 open-mythos）
│   ├── __init__.py        ← 公開匯出（只從 main.py 匯出；moda 不在內）
│   ├── main.py            ← 【正典】OpenMythos RDT（1085 行，使用者 import 的）
│   ├── moda.py            ← 【獨立實驗】MoDA + DeepSeek-MoE（1063 行，未匯出）
│   ├── variants.py        ← 7 個預設尺度：mythos_1b … mythos_1t
│   └── tokenizer.py       ← MythosTokenizer：HF AutoTokenizer 包裝（預設 gpt-oss-20b）
├── example.py             ← OpenMythos 煙霧測試（MLA，亂數輸入，印形狀）
├── examples/
│   ├── moda_example.py    ← MoDA 煙霧測試（含梯度檢查）
│   └── variants_example.py← 建一個 mythos_1b 並印參數量
├── tests/
│   ├── test_main.py       ← 【主測試】純 CPU 單元測試（678 行，無網路）
│   ├── test_rope_debug.py ← RoPE 視覺驗證腳本（直接跑，非 pytest）
│   ├── test_tokenizer.py  ← 分詞器測試（需網路下載 gpt-oss-20b）
│   ├── small_benchmark.py ← 【關鍵】OpenMythos vs baseline 的 loss + 深度外推比較
│   ├── bench_vs_transformer.py ← 效能基準（延遲、記憶體、深度縮放）
│   └── __init__.py        ← 空（0 bytes）
├── training/
│   ├── 3b_fine_web_edu.py ← FSDP + AdamW 預訓練腳本
│   └── requirements.txt   ← 訓練專屬相依（datasets>=3.6.0, loguru）
├── docs/
│   ├── open_mythos.md     ← OpenMythos 類別 API 參考
│   └── datasets.md        ← 推薦訓練資料集 + token 預算
├── pyproject.toml         ← 權威設定與相依
├── requirements.txt       ← 鬆散手動清單（勿信，用 pyproject）
├── README.md              ← 專案說明 + 理論背景
└── AGENTS.md              ← 給代理的工作指引（已知地雷全在這）
```

---

## 3. 兩套模型：OpenMythos vs MoDA

**這是新手最容易搞混的點**。倉庫有**兩套平行的模型實作**，互不相干：

| | OpenMythos（正典） | MoDA（實驗） |
|---|---|---|
| 檔案 | `open_mythos/main.py` | `open_mythos/moda.py` |
| 設定類別 | `MythosConfig` | `MoDAConfig`（`moda.py:59`） |
| 模型類別 | `OpenMythos` | `MoDAModel`（`moda.py:922`） |
| 被 `__init__.py` 匯出？ | ✅ 是 | ❌ 否（要手動 `from open_mythos.moda import ...`） |
| 核心機制 | 迴圈深度（共享權重反覆跑）+ LTI 穩定注入 | Mixture-of-Depths Attention（跨層 KV 在同一 softmax 注意）+ DeepSeek-MoE |
| 範例 | `example.py` | `examples/moda_example.py` |
| 測試 | `tests/test_main.py` | 無專屬測試（靠 `examples/moda_example.py` 煙霧測試） |

> 📌 **預設你用 OpenMythos**（`from open_mythos import OpenMythos`）。MoDA 是獨立的實驗分支，不接在 OpenMythos 上，別把兩者的 config 混用（`AGENTS.md` 明確警告）。

### MoDA 的獨特之處（供進階者參考）

- **MoDA Attention**（`moda.py:671-814`）：每個 query 同時注意「當前層的序列 KV（因果）」**和**「所有前面層在同一 token 位置的 depth KV」，在**單一 softmax** 下合併（`moda.py:801-803`）。這讓資訊能跨層直接流動，不需迴圈。
- **DeepSeek-V3 風格 MoE**（`moda.py:452-628`）：比 main.py 的 MoEFFN 更完整——支援 sigmoid gating、group-limited routing、專家級 balance loss（`moda.py:580-628`）。
- **post-norm**（`moda.py:896-901`），與 main.py 的 pre-norm 不同。

---

## 4. 小型確認（Small-Scale Verification）

這是本指南的核心：**如何用最少的時間/算力，確認這套程式碼「跑得起來、行為正確、理論成立」**。
按順序做，每一步驗證一個層次。

### 4.1 層次一：程式碼跑得起來（煙霧測試）

```bash
python example.py
```

**它做什麼**（`example.py` 全 50 行已讀）：
1. 建一個迷你 MLA 設定（`dim=256, vocab_size=1000, max_loop_iters=4`，`example.py:7-34`）
2. 餵**亂數 token**（`torch.randint`，`example.py:40`），跑前向，印 logits 形狀
3. 跑 `generate` 生成 8 token，印形狀
4. 印注入矩陣 A 的數值

**預期輸出**：參數量、`Logits shape: [2, 16, 1000]`、`Generated shape: [2, 24]`、譜半徑相關數值 < 1。

**這證明了什麼**：程式碼能跑、張量形狀正確、LTI 注入的 A 落在 (0,1)。
**這「不」證明**：模型會產生有意義的語言（它吃亂數、只看形狀）。

> ⚠️ `example.py:49` 印的「Spectral radius」用 `A.max().item()`，**標示有誤**（它只讀對角元素，不是真正的譜半徑）。在當前對角實作下數值恰好正確，但嚴格說應用 `torch.linalg.eigvals(A).abs().max()`（`README.md:99-103`，`AGENTS.md`）。

### 4.2 層次二：機械正確性（單元測試）

```bash
# 全部（純 CPU、快、無網路）
pytest

# 單一測試類別
pytest tests/test_main.py::TestLTIInjection

# 單一案例
pytest tests/test_main.py::TestLTIInjection::test_spectral_radius_stable_after_large_grad_step -v
```

**涵蓋什麼**（`tests/test_main.py` 678 行已讀）——用極小設定（`dim=64`，`test_main.py:28-57`）：

| 測試類別 | 驗證 | 關鍵斷言位置 |
|----------|------|-------------|
| `TestRMSNorm` | 正規化形狀、單位 RMS | `test_main.py:65-82` |
| `TestRoPE` / `TestRoPEExtended` | 旋轉保長、逆旋轉還原、相對位置性質、position 0 是恆等 | `test_main.py:90-251` |
| `TestGQAttention` | GQA 輸出形狀、KV 快取累積 | `test_main.py:259-287` |
| `TestMLAttention` | MLA 輸出形狀、快取存壓縮 c_kv | `test_main.py:295-329` |
| `TestMoEFFN` | MoE 形狀、router_bias 非梯度、共享專家恆啟動 | `test_main.py:354-375` |
| `TestLTIInjection` | **ρ(A)<1 且激進梯度（lr=1e3）後仍成立** | `test_main.py:471-489` |
| `TestACTHalting` | 停止機率在 [0,1] | `test_main.py:497-510` |
| `TestOpenMythosGQA/MLА` | 完整前向、生成、**深度外推**（n_loops=1 vs 3 輸出不同）、快取一致性 | `test_main.py:551-634` |
| `TestAttnTypeSwap` | MLA 快取比 GQA 小 | `test_main.py:642-674` |

> ⚠️ **三個測試檔不是 pytest**，直接用 `python` 跑：
> - `tests/small_benchmark.py`（見 4.3）
> - `tests/bench_vs_transformer.py`（見 4.5）
> - `tests/test_rope_debug.py`（見 4.7）
>
> ⚠️ `tests/test_tokenizer.py` **需網路**（下載 gpt-oss-20b），離線會卡住（見 4.8）。

### 4.3 層次三：Loss 會不會降（最關鍵的理論驗證）

這是驗證「架構真的能學」的實驗。`small_benchmark.py`（563 行已讀）在同樣資料上並排訓練 **OpenMythos vs vanilla transformer**，比較 loss。

```bash
# 預設：CPU、TinyStories、1000 步、batch 32、seq 256
python tests/small_benchmark.py

# GPU 加重
python tests/small_benchmark.py --device cuda --steps 5000 --batch-size 64 --seq-len 512
```

**關鍵設計**（讓比較公平）：
- 兩個模型用**同一個 MLA 設定**（`build_tiny_cfg`，`small_benchmark.py:271-300`：`dim=128, n_experts=4`）。
- Baseline 的層數 = `prelude + 1（一個迴圈區塊）+ coda`（`small_benchmark.py:427`），使**參數量相近**——這樣 delta 反映的是「迴圈架構」而非「參數多寡」。
- 兩者共享同一個 `TransformerBlock` 核心（`small_benchmark.py:84`），所以 delta 也不是核心差異。

**它量什麼**（`small_benchmark.py:11-19` 的 docstring）：
1. 每步訓練 loss + tokens/sec（兩模型）
2. 定期 held-out eval loss（`--eval-every`，預設每 200 步）
3. **深度外推掃描**（見 4.4）
4. 最後摘要表：initial/final/avg loss、wall-clock、tok/s、sec/step

**怎麼判讀結果**：
- **train loss 應該下降**——若不降或發散，表示這套架構在你的設定下訓不起來（紅燈）。
- OpenMythos 的 loss 應該降到接近 baseline（參數相近的前提下）——若是，綠燈。
- eval loss 的 `Δ` 欄會顯示兩者差距。

### 4.4 層次四：深度外推是否有效（RDT 最獨特的宣稱）

這是驗證「**訓練 N 個迴圈、測試 N+k 個迴圈仍然更好**」的實驗——RDT 的核心賣點。

`small_benchmark.py` 訓練結束後會自動跑深度掃描（`small_benchmark.py:540-559`）：
預設 `--depth-sweep 1,2,4,8,16`，模型訓練時用 `max_loop_iters=4`，然後在 eval 時分別用 1/2/4/8/16 個迴圈推理。

```bash
# 預設掃描
python tests/small_benchmark.py

# 加大外推範圍（訓練 4、測到 32）
python tests/small_benchmark.py --depth-sweep 1,2,4,8,16,32
```

**輸出長這樣**（`small_benchmark.py:551-559`）：
```
  OpenMythos (trained at n_loops=4):
        n_loops  eval loss  Δ vs trained
             1     X.XXXX      +0.XXXX
             2     X.XXXX      +0.XXXX
             4     X.XXXX                ←trained
             8     X.XXXX      -0.XXXX   ← 若為負=外推有效
            16     X.XXXX      -0.XXXX
```

**怎麼判讀**：
- 若 eval loss 在**超出訓練迴圈數（4）後繼續下降**（8、16 的 Δ 為負）→ **深度外推有效**，理論成立。
- 若超出後反而上升 → 過度思考（overthinking），或外推失敗。
- 這是**親自驗證 RDT 假說是否成立**的最直接實驗，repo 裡沒有附現成結果，要你自己跑。

### 4.5 層次五：效能基準（延遲與記憶體）

`bench_vs_transformer.py`（480 行已讀）量的是**速度與記憶體**，不是 loss。

```bash
# 小型 CPU/GPU 煙霧測試
python tests/bench_vs_transformer.py

# 指定規模與設備
python tests/bench_vs_transformer.py --size 1b --device cuda

# 自訂序列長度與迴圈數掃描
python tests/bench_vs_transformer.py --seq-lens 128,512,2048 --n-loops 1,4,8,16
```

**它量什麼**（`bench_vs_transformer.py:10-16`）：
- 參數數量（總 / MoE 啟動估算，`bench_vs_transformer.py:143-160`）
- Prefill 延遲 + 吞吐（多種序列長度，`:168-193`）
- Decode 延遲（KV 快取自迴歸，`:196-233`）
- 尖峰記憶體（僅 CUDA，`:119-122`）
- OpenMythos 深度縮放：延遲 vs n_loops（`:456-474`）

**可用參數**（`parse_args`，`:297-334`）：`--size small|1b`、`--device cuda|cpu`、`--dtype auto|fp32|bf16|fp16`、`--batch`、`--seq-lens`、`--n-loops`、`--decode-steps`、`--decode-prompt-len`。

> 📌 這支腳本用**亂數 token**（`bench_vs_transformer.py:177, 209, 219`），不訓練、不量 loss，純粹看「跑多快、吃多少記憶體」。

### 4.6 層次六：MoDA 模型煙霧測試

```bash
python examples/moda_example.py
```

**它做什麼**（`examples/moda_example.py` 73 行已讀）：
1. 建迷你 MoDA（`d_model=128, n_layers=4, 8 routed experts, top-2`，`:15-31`）
2. 餵亂數、跑前向 + loss（`:41-44`）
3. `loss.backward()`（`:46`）
4. **檢查梯度**：所有參數收到梯度（除了最後層的 write 投影，`:49-61`）
5. 抽查 MoE gate 與 depth write 投影的梯度範數（`:63-71`）

**這證明**：MoDA 的前向 + 反向都通、梯度流動正確（含跨層 depth read 的梯度）。

### 4.7 層次七：RoPE 視覺驗證

```bash
python tests/test_rope_debug.py
```

**它做什麼**（`test_rope_debug.py` 195 行已讀）——不是 pytest，是互動式除錯腳本，**印出中間張量**讓你親眼看：
1. 預計算頻率表（complex、magnitude、angle）
2. Position 0 是恆等相量（1+0j）
3. `apply_rope` 保形狀、保 dtype
4. **保長（isometry）**——旋轉不改範數
5. Position 0 是恆等變換
6. 不同位置產生不同旋轉
7. **逆旋轉還原原值**
8. `start_pos` 正確性（生成 bug 的修復驗證，`:140-163`）
9. **相對位置性質**：`<RoPE(q,m), RoPE(k,n)>` 只依賴 (n-m)

**用途**：懷疑 RoPE 有問題時，用這支肉眼看每個性質是否成立。

### 4.8 層次八：分詞器測試（需網路）

```bash
pytest tests/test_tokenizer.py -v -s
```

> ⚠️ **會下載 `openai/gpt-oss-20b`**（`tokenizer.py:3, 30` 的 `AutoTokenizer.from_pretrained`）。**離線會失敗/卡住**。

**測什麼**（`test_tokenizer.py` 75 行已讀）：載入分詞器、vocab_size、encode/decode roundtrip、空字串、長文、自訂 model_id、vocab_size 一致性。

### 小型確認的優先順序建議

| 你想確認 | 跑什麼 | 耗時 |
|----------|--------|------|
| 裝對了 | 1.3 的 import 檢查 | 秒 |
| 程式跑得起來 | 4.1 `example.py` | 秒 |
| 機械正確 | 4.2 `pytest` | ~30 秒 |
| **loss 會降**（理論可學） | 4.3 `small_benchmark.py` | CPU 數分鐘；GPU 更快 |
| **深度外推有效** | 4.4（同上，看結尾掃描表） | 同上 |
| 跑多快 | 4.5 `bench_vs_transformer.py` | 分鐘 |
| MoDA 通 | 4.6 `moda_example.py` | 秒 |
| RoPE 對 | 4.7 `test_rope_debug.py` | 秒 |

---

## 5. 完整訓練

> ⚠️ 訓練相依與主程式**不同**：`training/requirements.txt` 需 `datasets>=3.6.0`（主程式 `>=2.18`）與 `loguru`。**分開安裝**（`AGENTS.md`）。

```bash
# 安裝訓練專屬相依
pip install -r training/requirements.txt

# 單 GPU
python training/3b_fine_web_edu.py

# 多 GPU（自動偵測 GPU 數）
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py
```

**關鍵設定**（`training/3b_fine_web_edu.py:374-386`）：seq_len=2048、目標 30B tokens、AdamW lr=3e-4 cosine、bf16、FineWeb-Edu `sample-10BT`。

**推薦資料集**（`docs/datasets.md`）：FineWeb-Edu（主）、OpenHermes 2.5（指令，混 5%）、OpenWebMath（數學）。Token 預算表見 `docs/datasets.md:36-41`。

---

## 6. 已知地雷與限制

### 6.1 程式碼層

| 地雷 | 出處 | 影響 |
|------|------|------|
| `load_tokenizer`/`get_vocab_size` 列在 `__all__` 但未匯入 | `__init__.py:52-53` | import 即崩 |
| `example.py` 的「譜半徑」用 `A.max()` 非 eigvals | `example.py:49` | 標示錯誤（對角實作下數值碰巧對） |
| README 的 `mythos_7b()` 不存在 | `README.md:124` | 真實變體只有 1b/3b/10b/50b/100b/500b/1t |
| 根目錄 `requirements.txt` 與 pyproject 衝突 | `requirements.txt:1` vs `pyproject.toml:41` | 以 pyproject 為準 |
| 訓練相依與主程式不同 | `training/requirements.txt` | 需分開裝 |
| MoE 派工是 Python 雙層迴圈 | `main.py:520-527` | 量產規模極慢，需 grouped GEMM 重寫 |

### 6.2 理論層（最重要的一點）

**這個 repo 沒有訓練好的模型**——沒有 checkpoint、沒有發布的權重、沒有 benchmark 結果。
`example.py` 餵亂數只看形狀；`test_main.py` 測機械正確性不測品質；`small_benchmark.py` 是**讓你產生結果的工具**，不是已附的結果。

README 第 30 行明說：**"theoretical reconstruction ... based solely on publicly available research and speculation"**。

所以：**「跑得起來」≠「是個能用的模型」**。要回答「能不能用」，必須自己跑完 `small_benchmark.py` 看 loss 降不降、深度外推有沒有效——那才是這套理論成立與否的證據。

---

## 附錄：指令速查

```bash
# 安裝
pip install -e .

# 煙霧測試
python example.py                          # OpenMythos（MLA）
python examples/moda_example.py            # MoDA
python examples/variants_example.py        # 印 1B 參數量

# 測試
pytest                                      # 全部單元測試
pytest tests/test_main.py::TestLTIInjection::test_spectral_radius_stable_after_large_grad_step -v
python tests/test_rope_debug.py            # RoPE 視覺驗證（非 pytest）
pytest tests/test_tokenizer.py -v -s        # 需網路

# loss / 深度外推（最關鍵）
python tests/small_benchmark.py
python tests/small_benchmark.py --depth-sweep 1,2,4,8,16,32

# 效能
python tests/bench_vs_transformer.py
python tests/bench_vs_transformer.py --size 1b --device cuda

# 訓練
pip install -r training/requirements.txt
python training/3b_fine_web_edu.py

# 品質
ruff check .
black --check .
```
