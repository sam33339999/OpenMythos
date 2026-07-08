# 14 — Fusion 參數、預設與遞迴保護

> 承接第 13 篇。上一篇談結構化分析與降級；本文補完 Fusion 的**參數全貌**、**preset 體系**，
> 以及一個與 Mythos 截然對比的設計決定——**遞迴保護（recursion guard）**。
> Mythos 鼓勵「無限加迴圈」，Fusion 卻**禁止** panel/judge 再呼叫自己。這個反差是第 15 篇典範對比的引子。

## 1. 參數全貌（Server Tool 形式）

把三份文件的參數表合併，Server Tool 可設的完整欄位
（[fusion-router](https://openrouter.ai/docs/guides/routing/routers/fusion-router) 參數表為最完整版本）：

| 欄位 | 預設 | 範圍／型別 | 說明 |
|------|------|-----------|------|
| `analysis_models` | Quality preset 三模型（`~anthropic/claude-opus-latest`, `~openai/gpt-latest`, `~google/gemini-pro-latest`） | 1–8 個 slug | panel 組成；每個帶 web_search+web_fetch |
| `model` | 你的 outer model | slug | judge；用 `openrouter/fusion` 時也是寫最終答案的模型 |
| `max_tool_calls` | 8 | 1–16 | panel/judge 在 web 工具迴圈內最多幾步 |
| `max_completion_tokens` | provider 預設 | int | 每 panel/judge 呼叫最大輸出（含推理） |
| `reasoning` | provider 預設 | `{effort, max_tokens}` | 轉發給 panel/judge 的推理設定 |
| `temperature` | provider 預設 | 0–2 | 轉發給 panel；**judge 恆為 0** |

### 兩個值得記住的隱性規則

1. **judge 恆為 temperature 0**（[fusion-router](https://openrouter.ai/docs/guides/routing/routers/fusion-router)）。
   你設的 `temperature` 只影響 panel，不影響 judge。比對必須可重現。
2. **web 工具兩端都啟用**（panel + judge），不是只有 panel 能查網路（[server-tools](https://openrouter.ai/docs/guides/features/server-tools/fusion)）。

## 2. Preset 體系

不想挑模型？用 preset slug，panel 與 judge 由官方代選
（[plugin 文件](https://openrouter.ai/docs/guides/features/plugins/fusion) 的「Presets」）：

```json
{ "model": "openrouter/fusion", "plugins": [{ "id": "fusion", "preset": "general-budget" }] }
```

slug 命名 `<task>-<tier>`：

| Preset | 定位 |
|--------|------|
| `general-high` | 最強全用途 panel |
| `general-budget` | 便宜 panel + 前沿 judge（低成本仍強綜合） |
| `general-fast` | 延遲同質 panel（TTFT 相近，無單一模型拖慢 fan-out）+ 前沿 judge |

關鍵規則：**顯式 `analysis_models` 或 `model` 永遠蓋過 preset**（[plugin 文件](https://openrouter.ai/docs/guides/features/plugins/fusion)）。
所以 preset 是「懶人預設」，不是硬性綁定。

### `general-fast` 與 `fusion-flash` 的關係

`openrouter/fusion-flash` 是獨立列出的 model，預先釘了 `general-fast` preset
（[fusion-router 文件](https://openrouter.ai/docs/guides/routing/routers/fusion-router) 的「Fast preset model」）。
兩者等價：

```json
{ "model": "openrouter/fusion-flash", ... }
≡
{ "model": "openrouter/fusion", "plugins": [{ "id": "fusion", "preset": "general-fast" }], ... }
```

差別只是 `fusion-flash` 在 `/api/v1/models` 有獨立條目、用量歸因分開。你顯式傳的設定永遠贏過這個隱含 preset。

## 3. 成本模型

Fusion 的成本是線性的（[fusion-router 文件](https://openrouter.ai/docs/guides/routing/routers/fusion-router) 的「Cost」）：

> Fusion runs N panel calls + 1 judge call in addition to your normal request. With the default 3-model panel,
> expect roughly 4–5× the cost of a single completion on the same prompt. Cost scales linearly with panel size.

拆解：1（outer）+ N（panel）+ 1（judge）。預設 N=3，所以約 5 次完成呼叫 ≈ 4–5× 單次成本。
要省錢就縮小 panel（降到 1–2 個）或用 `general-budget` preset。

> 📌 對照 Mythos：Mythos 加迴圈的邊際成本是「同一組權重多跑一次前向」，**參數不增長**（`README.md:259-265`）。
> Fusion 加 panel 模型的成本是「多呼叫一個完整模型」，**線性增加**。
> 這是兩種 test-time scaling 在「成本曲線」上的根本差異——第 18 篇會深入。

## 4. 遞迴保護：Fusion 為何是單層

這是本篇最重要的設計決定。三份文件都重複同一句
（[server-tools](https://openrouter.ai/docs/guides/features/server-tools/fusion)、[plugin](https://openrouter.ai/docs/guides/features/plugins/fusion)、[fusion-router](https://openrouter.ai/docs/guides/routing/routers/fusion-router)）：

> Inner fusion calls carry an `x-openrouter-fusion-depth` header. Panel and judge models cannot recursively
> invoke `openrouter:fusion` — the plugin refuses to inject the tool a second time, keeping deliberation
> bounded to a single level.

含義：

1. 每次 fusion 呼叫帶一個 `x-openrouter-fusion-depth` header 標記深度。
2. panel 與 judge 模型**不能再呼叫 `openrouter:fusion`**——plugin 在注入工具時檢查這個 header，
   若已是內層呼叫就**拒絕第二次注入**。
3. 結果：審議被**限制在單一層級**。

### 為什麼要禁止遞迴

若允許遞迴，會出現組合爆炸：每個 panel 模型再開一個 panel（各 3 個）、每個子 panel 再開……
深度 d、分支 N 的樹會產生 O(N^d) 次呼叫，成本與延遲瞬間失控。
`fusion_invocation_capped`（第 13 篇的 failure_reason 之一）就是這個保護的具體表現——
同一輪第二次呼叫會被拒。

## 5. 與 Mythos 迴圈的最尖銳對比

把遞迴保護放回 Part I 的脈絡，反差極為鮮明：

| 設計選擇 | OpenMythos | OpenRouter Fusion |
|----------|-----------|-------------------|
| 對「重複」的態度 | **鼓勵**：迴圈是核心，越多越深 | **限制**：審議只允許單層 |
| 深度上限 | 推理期可超過訓練值（深度外推，第 10 篇） | 硬性 1 層（header 守護） |
| 為何如此 | 同一組權重反覆，邊際成本是計算量，可控；且有 LTI 保證收斂（第 05 篇） | 每層是完整模型呼叫，成本隨分支指數成長，必須設界 |
| 防發散機制 | ρ(A)<1 構造保證 | recursion guard + capped invocation |

換句話說：**Mythos 的「重複」在連續空間、成本可控、有穩定性保證，所以可以深；
Fusion 的「重複」在離散呼叫空間、成本指數成長、無收斂保證，所以必須淺。**

這不是「誰對誰錯」，而是**兩種機制層級**各自合理的工程選擇。第 15 篇會把這個對比系統化。

## 6. 一個可遷移的設計原則：為「重複」設界

Fusion 的遞迴保護帶來一個通用啟示（餵養第 21 篇）：

> 任何允許「重複／遞迴」的機制，都必須顯式回答「重複到哪裡停」。
> - 若重複在連續空間、成本可控：靠穩定性條件（如 ρ<1）自然收斂，可讓它深。
> - 若重複在離散空間、成本不可控：靠硬性界（如 depth header + capped）強制停止，必須淺。

Mythos 與 Fusion 分別是這兩種答案的範例。設計自己的回授 LLM 時，這個抉擇是第一個要做的架構決策。

## 7. 本文結論與下篇預告

- 參數全貌：analysis_models、model、max_tool_calls、max_completion_tokens、reasoning、temperature。
- judge 恆 temperature 0；web 工具兩端啟用。
- preset 體系（high/budget/fast），顯式設定永遠蓋過 preset；`fusion-flash` ≡ `general-fast` 預設。
- 成本線性：1 outer + N panel + 1 judge；預設約 4–5×。
- **遞迴保護**：`x-openrouter-fusion-depth` header + 拒絕二次注入，審議限單層。
- 設計原則：重複機制必須顯式設界——連續空間靠穩定性收斂、離散空間靠硬性界。

**下一篇（15）** 是 Part II 壓軸：把前四篇的 Fusion 與 Part I 的 Mythos 做系統化的**典範對比**，
確立「單模型反覆運算 vs 多模型並行審議」這組對立，作為 Part III（理論與整合）的橋梁。

## 引用證據（OpenRouter 官方文件，已 webfetch 確認 HTTP 200）

- 參數全貌與 judge temperature 0：https://openrouter.ai/docs/guides/routing/routers/fusion-router
- preset 體系與命名：https://openrouter.ai/docs/guides/features/plugins/fusion
- fusion-flash 等價性：https://openrouter.ai/docs/guides/routing/routers/fusion-router
- 成本線性：https://openrouter.ai/docs/guides/routing/routers/fusion-router
- 遞迴保護（三處）：https://openrouter.ai/docs/guides/features/server-tools/fusion ， https://openrouter.ai/docs/guides/features/plugins/fusion ， https://openrouter.ai/docs/guides/routing/routers/fusion-router
- Mythos 成本曲線對照：`README.md:259-265`（第 03、10 篇）
