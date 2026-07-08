# 11 — OpenRouter Fusion 總覽

> Part II 首篇。Part I 用十篇拆解了 OpenMythos 的「單模型反覆運算」；從本篇起轉向一個截然不同的機制——
> **OpenRouter Fusion**：把同一個問題丟給一組模型並行回答，再由一個判斷模型比對、收斂成結構化分析，
> 最後由你的模型寫出最終答案。本文建立 Fusion 的整體心智模型，作為第 12–15 篇的基礎。

## 1. Fusion 是什麼

OpenRouter 官方文件對 Fusion 的定義（已實際抓取確認，來源：
[server-tools/fusion](https://openrouter.ai/docs/guides/features/server-tools/fusion)）：

> The `openrouter:fusion` server tool gives any model access to **multi-model deliberation**.
> When your model decides a prompt benefits from multiple perspectives, it invokes this tool —
> a panel of models answers in parallel, a judge compares their responses, and the structured
> analysis comes back to your model for the final answer.

也就是說，Fusion 是一個**伺服器端工具**（server tool），任何模型都可以呼叫它。
被呼叫時的流程（[fusion-router 文件](https://openrouter.ai/docs/guides/routing/routers/fusion-router)的 mermaid 圖）：

```
你的請求 → 你的模型 → (呼叫 openrouter:fusion) → Panel：最多 8 個模型並行 + web_search + web_fetch
                                                → Judge：比對回應 + web_search，產出結構化 JSON
                   ← 分析結果 ←─────────────────────
        → 最終答案
```

## 2. 五階段管線

把上述拆成五步（[plugin/fusion 文件](https://openrouter.ai/docs/guides/features/plugins/fusion)的編號）：

1. **注入工具**：若用 `model: "openrouter/fusion"` 別名，router 會把別名解析成真實模型並掛上 `openrouter:fusion` 工具。
2. **模型決定是否呼叫**：你的模型讀完 prompt，自己判斷這題值不值得「多方視角」。
3. **Panel 並行回答**：一組模型（panel）同時回答你的 prompt，每個都帶 `openrouter:web_search` 與 `openrouter:web_fetch`。
4. **Judge 比對（不融合）**：judge 收到所有 panel 回應，同樣帶 web 工具，**比對而非合併**它們，
   產出結構化 JSON（consensus、contradictions、unique_insights、blind_spots）。
5. **你的模型寫最終答案**：拿到結構化分析後，由你的模型（outer model）撰寫最終回覆。

> ⚠️ 一個重要的語意區別（[plugin 文件](https://openrouter.ai/docs/guides/features/plugins/fusion)）：
> judge 是 **compares, not merges**（比對，而非合併）。它不做加權平均或多數決；
> 它把「多數模型同意的」當高信心共識、標出分歧、保留單一模型的獨特洞見、點出無人涵蓋的盲區。
> 最終答案由你的模型基於這份分析寫出，所以「結果不是簡單多數決」。

## 3. 結構化分析 JSON 的長相

成功時工具回傳（[server-tools 文件](https://openrouter.ai/docs/guides/features/server-tools/fusion)）：

```json
{
  "status": "ok",
  "analysis": {
    "consensus": ["所有或多數 panel 模型同意的論點"],
    "contradictions": [
      { "topic": "...", "stances": [{ "model": "...", "stance": "..." }] }
    ],
    "partial_coverage": [
      { "models": ["..."], "point": "只有部分模型提到" }
    ],
    "unique_insights": [
      { "model": "...", "insight": "只有一個模型提出的洞見" }
    ],
    "blind_spots": ["沒有任何 panel 模型涵蓋的主題"]
  },
  "responses": [
    { "model": "anthropic/claude-opus-4.5", "content": "..." },
    { "model": "openai/gpt-4.1", "content": "..." }
  ]
}
```

五個分析欄位各自有明確語意——這份結構是第 13 篇的主題。本文先建立整體印象：Fusion 的產出是
**結構化的、可比對的、附帶原始 panel 回應的**，不是一段糊在一起的文字。

## 4. 與 OpenMythos 迴圈的初步對照（為第 15 篇鋪路）

把 Fusion 放在 Part I 的對照框裡，差異立刻浮現：

| 維度 | OpenMythos 迴圈 | OpenRouter Fusion |
|------|-----------------|-------------------|
| 誰在反覆 | 同一個模型，同一組權重 | 多個不同模型，各跑一次 |
| 「深度」來源 | 迴圈次數 T | panel 模型數 N |
| 「整合」發生在哪 | 每個迴圈的 `A·h + B·e` 注入（連續空間） | judge 一次性比對（符號空間） |
| 中間產物 | 連續隱藏狀態（不可讀） | 多段文字回應 + 結構化 JSON（可讀） |
| 時序 | 序列（一步接一步） | 並行（panel 同時跑）再序列（judge） |
| 成本軸 | 計算隨 T 線性長 | 呼叫數隨 N 線性長（N panel + 1 judge + 1 outer） |

兩者都在做「把多個視角收斂成一個答案」，但**機制層級完全不同**：
Mythos 在單一模型的潛在空間裡反覆；Fusion 在多個模型之間透過自然語言協調。
這個對照是本系列第 15 篇的核心，也會在第 17 篇進一步映射到「注入 vs 整合」。

## 5. 為什麼說 Fusion 是「整合 LLM 的最後一步」

呼應本系列研究目標中對 Fusion 的描述——「不同 LLM 的回應一起整並，最後透過一個 LLM 進行把最終結果整合」——
Fusion 的設計**正好**對應這個描述，而且做了兩個關鍵強化：

1. **整合不是平均，是比對**：judge 不把回應糊成一團，而是分類成共識／矛盾／洞見／盲區。
   這比「取眾數」或「拼接」保留了更多資訊，讓 outer 模型能做有根據的取捨。
2. **整合後仍由 LLM 收尾**：結構化分析不是最終答案，而是餵給 outer 模型的「證據摘要」。
   最終答案的措辭、取捨、風格仍由一個 LLM 完成——這正是「透過一個 LLM 進行整合」的精確實作。

## 6. 何時該用 Fusion

官方明確的指引（[server-tools 文件](https://openrouter.ai/docs/guides/features/server-tools/fusion)）：

> The tool's description tells the model to call `openrouter:fusion` only when a task genuinely
> benefits from multiple perspectives — research questions, multi-domain critique, "compare and
> contrast" prompts, or anything where being wrong is expensive. Simple tactical prompts won't trigger it.

也就是：Fusion 為「研究問題、跨領域批判、比較型任務、出錯成本高」的場景設計。
短小戰術型 prompt 不會觸發它（模型自己會判斷不值得）。要強制每次都跑，可設 `tool_choice: "required"`
（第 12、14 篇詳述）。

## 7. 本文結論與下篇預告

- Fusion = panel（多模型並行）+ judge（結構化比對）+ outer model（寫最終答案）。
- judge **比對而非合併**，產出 consensus／contradictions／insights／blind_spots。
- 它對應「把多個 LLM 回應整並、最後由一個 LLM 整合」的精確實作，且比樸素合併更保留資訊。
- 與 Mythos 迴圈是「機制層級不同」的兩種收斂方式。

**下一篇（12）** 解析 Fusion 的**三種使用入口**——Router 別名、Plugin、Server Tool——
它們打的是同一條管線，但控制粒度與使用場景不同。選對入口是實務整合的第一步。

## 引用證據（OpenRouter 官方文件，已 webfetch 確認 HTTP 200）

- Server tool 定義與五階段：https://openrouter.ai/docs/guides/features/server-tools/fusion
- Plugin 編號流程與「compares not merges」：https://openrouter.ai/docs/guides/features/plugins/fusion
- Fusion Router mermaid 與說明：https://openrouter.ai/docs/guides/routing/routers/fusion-router
- 結構化分析 JSON：https://openrouter.ai/docs/guides/features/server-tools/fusion
- 觸發時機指引：https://openrouter.ai/docs/guides/features/server-tools/fusion
