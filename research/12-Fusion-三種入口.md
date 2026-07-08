# 12 — Fusion 的三種入口：Router、Plugin、Server Tool

> 承接第 11 篇。上一篇建立 Fusion 的整體管線；本文解析官方提供的**三種使用入口**。
> 官方明言「三者打的是同一條 pipeline」（[plugin 文件](https://openrouter.ai/docs/guides/features/plugins/fusion)），
> 但控制粒度、使用姿態、適用場景各異。選對入口是落地整合的第一步。

## 1. 三入口總覽

| 入口 | 形式 | 一句話定位 | 控制粒度 |
|------|------|-----------|----------|
| **Fusion Router** | `model: "openrouter/fusion"`（model slug） | 把整條管線當成一個「模型」用 | 最省事，工具自動注入 |
| **Fusion Plugin** | `plugins: [{id: "fusion", ...}]` | 在已有 model 上掛「設定面板」 | 中等，可選 preset 與模型 |
| **Server Tool** | `tools: [{type: "openrouter:fusion", ...}]` | 直接宣告工具，模型自行決定何時呼叫 | 最高，可搭配其他工具 |

三者的等價性（[fusion-router 文件](https://openrouter.ai/docs/guides/routing/routers/fusion-router)）：

> `openrouter/fusion` is equivalent to enabling the `openrouter:fusion` server tool on the configured model.
> These behave identically.

## 2. 入口一：Fusion Router（model slug）

最簡單。把 `model` 設成 `"openrouter/fusion"`，router 會：
1. 把別名解析成一個真實模型（預設 Quality preset 的第一個，`~anthropic/claude-opus-latest`）。
2. 自動注入 `openrouter:fusion` 工具。

```json
{
  "model": "openrouter/fusion",
  "messages": [{ "role": "user", "content": "...你的問題..." }]
}
```

([fusion-router 文件](https://openrouter.ai/docs/guides/routing/routers/fusion-router) 的「Model alias」範例)

### 快速預設：`openrouter/fusion-flash`

另有一個獨立列出的模型 `openrouter/fusion-flash`，預先選了 `general-fast` preset——
一個「延遲同質」panel（每個模型 TTFT 相近，沒有單一模型拖慢 fan-out），適合快速 agent 輪次
（[fusion-router 文件](https://openrouter.ai/docs/guides/routing/routers/fusion-router) 的「Fast preset model」一節）。

變體後綴（`:free`、`:nitro` 等）在 `openrouter/fusion` 與 `openrouter/fusion-flash` 上解析方式相同。

## 3. 入口二：Fusion Plugin

當你不想換模型、只想在現有 model 上「附加」Fusion 能力，用 plugin。Plugin 是 server tool 的**設定介面**
（[plugin 文件](https://openrouter.ai/docs/guides/features/plugins/fusion)）：

> The Fusion plugin is a configuration surface for the `openrouter:fusion` server tool.

```json
{
  "model": "openrouter/fusion",
  "plugins": [
    {
      "id": "fusion",
      "analysis_models": ["~anthropic/claude-opus-latest", "~openai/gpt-latest"],
      "model": "~openai/gpt-latest"
    }
  ],
  "messages": [...]
}
```

Plugin 可設的欄位（[plugin 文件](https://openrouter.ai/docs/guides/features/plugins/fusion) 的表格）：

| 欄位 | 預設 | 說明 |
|------|------|------|
| `preset` | 無 | 預設 preset slug（如 `general-high`），展開成 panel+judge；顯式 `analysis_models`/`model` 覆蓋它 |
| `analysis_models` | Quality preset 三模型 | panel 組成；每個帶 web_search+web_fetch；1–8 個 |
| `model` | Quality preset 首模型 | judge 模型；用 `openrouter/fusion` 時也是寫最終答案的模型 |
| `max_tool_calls` | 8 | panel/judge 在 web 工具迴圈內最多幾步才須產出文字；範圍 1–16 |
| `enabled` | true | 設 false 可對單一請求繞過 fusion |

### Preset slug 命名

`<task>-<tier>`：task 是優化方向，tier 是品質／成本／速度取捨
（[plugin 文件](https://openrouter.ai/docs/guides/features/plugins/fusion) 的「Presets」）：

| Preset | 用途 |
|--------|------|
| `general-high` | 最強全用途 panel |
| `general-budget` | 較便宜 panel + 前沿 judge |
| `general-fast` | 延遲同質 panel + 前沿 judge，適合快速 agent 輪次 |

顯式 `analysis_models` 或 `model` 永遠蓋過 preset。

## 4. 入口三：Server Tool（直接宣告）

控制最細。你自己選 outer model，把 `openrouter:fusion` 當成一個工具宣告，模型自行判斷何時呼叫：

```json
{
  "model": "~anthropic/claude-opus-latest",
  "messages": [...],
  "tools": [{ "type": "openrouter:fusion" }]
}
```

([server-tools 文件](https://openrouter.ai/docs/guides/features/server-tools/fusion) 的 Quick start)

要客製 panel 與 judge，在工具上加 `parameters`（[server-tools 文件](https://openrouter.ai/docs/guides/features/server-tools/fusion) 的 Parameters 表）：

```json
"tools": [{
  "type": "openrouter:fusion",
  "parameters": {
    "analysis_models": ["~google/gemini-flash-latest", "deepseek/deepseek-v3.2", "~moonshotai/kimi-latest"],
    "model": "~anthropic/claude-opus-latest"
  }
}]
```

Server tool 可設的欄位比 plugin 多出幾個（[fusion-router 文件](https://openrouter.ai/docs/guides/routing/routers/fusion-router) 的參數表）：

| 欄位 | 預設 | 說明 |
|------|------|------|
| `analysis_models` | Quality preset | panel，1–8 個 |
| `model` | outer model | judge |
| `max_tool_calls` | 8 | web 工具迴圈上限；1–16 |
| `max_completion_tokens` | provider 預設 | 每 panel/judge 呼叫的最大輸出 token（含推理），防止推理重模型耗盡預算 |
| `reasoning` | provider 預設 | 轉發給 panel/judge 的推理設定（`effort`、`max_tokens`） |
| `temperature` | provider 預設 | 轉發給 panel 的溫度（0–2）；**judge 恆為 0** |

> 📌 「judge 恆為 temperature 0」是個值得記住的設計——比對必須可重現、不摻隨機。
> 這與 OpenMythos 的 LTI 注入「構造性穩定」有異曲同工之妙：兩者都把「該確定的部分釘死」。

## 5. 三者的關鍵差異：誰觸發、誰收尾

| 問題 | Router 別名 | Plugin | Server Tool |
|------|------------|--------|-------------|
| 工具誰注入 | router 自動 | plugin 注入 | 你手動宣告 |
| outer model 誰定 | 預設 Quality preset 首模型 | 你定（`model` 欄位） | 你定（request 的 `model`） |
| 能否混其他工具 | 否（只注入 fusion 一個） | 視 request | **可以**（`tools` 陣列可加別的工具） |
| 適合場景 | 一次到位、不想挑模型 | 想客製 panel/judge 但仍用別名 | 要 fusion 與其他工具協同 |

最後一點是 server tool 的獨特優勢：因為 `tools` 是陣列，你可以讓同一個模型同時擁有
fusion、web_search、自訂函式呼叫等工具，由模型自己調度。Router 別名只注入 fusion 一個工具，
所以「要求某個工具」(`tool_choice: "required"`) 等同強制 fusion（[fusion-router 文件](https://openrouter.ai/docs/guides/routing/routers/fusion-router)）。

## 6. 強制每次都跑：`tool_choice: "required"`

預設由模型決定是否呼叫。要保證每次都跑：

```json
{ "model": "openrouter/fusion", "tool_choice": "required", "messages": [...] }
```

([fusion-router 文件](https://openrouter.ai/docs/guides/routing/routers/fusion-router) 的「Forcing fusion」)
因為 `openrouter/fusion` 只注入一個工具，「要求某個工具呼叫」就等同強制 fusion。
但若 request 還含其他工具，模型可能選別的。

## 7. 回應裡的 router 欄位

回應的 `model` 欄位回報**實際**處理請求的模型，不是 `openrouter/fusion` 別名
（[fusion-router 文件](https://openrouter.ai/docs/guides/routing/routers/fusion-router)）：

```json
{ "id": "gen-...", "model": "anthropic/claude-opus-4.5", ... }
```

要確認是否走了 Fusion，查 generation metadata 的 `router` 欄位：

```json
{ "data": { "id": "gen-...", "model": "anthropic/claude-opus-4.5", "router": "openrouter/fusion" } }
```

## 8. 本文結論與下篇預告

- 三入口同一管線，差在控制粒度：Router 最省、Plugin 中等、Server Tool 最細。
- Server Tool 獨特處是能與其他工具並存，由模型自行調度。
- Plugin/Server Tool 的參數能客製 panel、judge、token 上限、推理設定；judge 恆為 temperature 0。
- `tool_choice: "required"` 強制每次都跑。
- 回應的 `model` 是實際模型，要查 `router` 欄位才知是否走 Fusion。

**下一篇（13）** 深入 Fusion 產出的核心——結構化分析 JSON 的五個欄位語意，以及 judge 失敗時的
降級處理（graceful degradation）。這份「可比對的結構」正是 Fusion 比樸素合併更強的關鍵。

## 引用證據（OpenRouter 官方文件，已 webfetch 確認 HTTP 200）

- 三入口等價性：https://openrouter.ai/docs/guides/features/plugins/fusion ， https://openrouter.ai/docs/guides/routing/routers/fusion-router
- Router 別名與 fusion-flash：https://openrouter.ai/docs/guides/routing/routers/fusion-router
- Plugin 設定介面與 preset：https://openrouter.ai/docs/guides/features/plugins/fusion
- Server tool 與 parameters：https://openrouter.ai/docs/guides/features/server-tools/fusion
- judge 恆 temperature 0：https://openrouter.ai/docs/guides/routing/routers/fusion-router 參數表
- tool_choice 與 router 欄位：https://openrouter.ai/docs/guides/routing/routers/fusion-router
