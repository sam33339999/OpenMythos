# 13 — Fusion 的結構化分析與判斷模型

> 承接第 12 篇。上一篇談三種入口；本文鑽進 Fusion 真正的「產品力」所在——
> judge 產出的**結構化分析 JSON**，以及當 panel 或 judge 失敗時的**降級處理**。
> 這份結構是 Fusion 比樸素「把多段回應拼接」更強的根本原因，也直接啟發第 17 篇的「整合 vs 注入」對照。

## 1. 結構化分析的五個欄位

成功時，工具回傳的 `analysis` 物件（[server-tools 文件](https://openrouter.ai/docs/guides/features/server-tools/fusion)）：

```json
"analysis": {
  "consensus":        ["所有或多數 panel 模型同意的論點"],
  "contradictions":   [{ "topic": "...", "stances": [{ "model": "...", "stance": "..." }] }],
  "partial_coverage": [{ "models": ["..."], "point": "只有部分模型提到" }],
  "unique_insights":  [{ "model": "...", "insight": "只有一個模型提出" }],
  "blind_spots":      ["沒有任何 panel 模型涵蓋的主題"]
}
```

逐欄位的語意（綜合 [server-tools](https://openrouter.ai/docs/guides/features/server-tools/fusion) 與 [fusion-router](https://openrouter.ai/docs/guides/routing/routers/fusion-router) 文件）：

| 欄位 | 語意 | 對 outer 模型的用處 |
|------|------|-------------------|
| `consensus` | 多數模型一致的論點，**視為高信心** | 可放心採信、寫進答案 |
| `contradictions` | 模型間的分歧，附帶各方立場與出處 | 需 outer 模型裁決或並陳 |
| `partial_coverage` | 只有部分模型涵蓋的點 | 提示某論點證據不足 |
| `unique_insights` | 單一模型的獨特洞見 | 值得考量但未必代表共識 |
| `blind_spots` | 全體 panel 都沒談的主題 | 提示答案可能的缺口 |

這套分類的設計意圖（[fusion-router 文件](https://openrouter.ai/docs/guides/routing/routers/fusion-router)）：

> the judge compares the panel responses rather than merging them: it treats what all or most models
> agree on as higher-confidence consensus, surfaces contradictions, preserves unique insights from
> individual models, and flags blind spots none of them addressed.

關鍵字是 **compares rather than merges**：judge 不把回應糊在一起，而是**分類、標註出處、保留分歧**。
這讓 outer 模型拿到的是「附帶可信度標籤的證據集」，而非「失真的平均」。

## 2. 為什麼這比樸素合併強

設想三種樸素的「多回應整合」：

| 樸素法 | 問題 |
|--------|------|
| 多數決（取最多模型說的） | 少數正確的會被埋沒；`unique_insights` 正是為了搶救這個 |
| 加權平均文字 | 自然語言無法「平均」；語意會崩壞 |
| 全部拼接 | 冗長、自相矛盾、outer 模型難以裁決 |

Fusion 的結構化分析把「一致 / 分歧 / 獨特 / 缺漏」拆開，等於給 outer 模型一份**已標註可信度的決策摘要**。
這也解釋了為什麼最終答案仍由一個 LLM 寫——因為「如何在共識與矛盾間取捨」需要判斷力，不是機械式合併能做的。

## 3. 降級處理：當東西出錯時

Fusion 的容錯設計是它最值得借鏡的工程細節之一（[server-tools 文件](https://openrouter.ai/docs/guides/features/server-tools/fusion)）。

### 部分 panel 失敗

當**部分** panel 模型出錯但至少一個成功：結果仍 `status: "ok"`，並附 `failed_models` 陣列說明誰失敗、為何。
outer 模型可基於成功的回應繼續。

### Judge 失敗（degradation）

當 **panel 成功但 judge 失敗**（上游錯誤、空回應、或產出不是合法分析 JSON）：工具**不**報錯，
而是回 `status: "ok"` 並**省略** `analysis`，只給原始 panel `responses`：

```json
{
  "status": "ok",
  "responses": [{ "model": "anthropic/claude-opus-4.5", "content": "..." }]
}
```

([server-tools 文件](https://openrouter.ai/docs/guides/features/server-tools/fusion) 的「Judge degradation」)

含義：即使 judge 掛了，outer 模型仍能從原始 panel 回應自己綜合答案——**降級而非中斷**。

### 硬失敗

只有當「無法產出任何有用輸出」時才回 `status: "error"`，並附帶型別化的 `failure_reason`
（[server-tools 文件](https://openrouter.ai/docs/guides/features/server-tools/fusion) 的「Hard failures」）：

| failure_reason | 含義 |
|----------------|------|
| `all_panels_failed` | 每個 panel 模型都出錯 |
| `insufficient_credits` | 全失敗且至少一個是額度不足 |
| `rate_limited` | 全失敗且至少一個被限流 |
| `fusion_invocation_capped` | 同一輪已呼叫過 fusion，第二次被拒 |
| `unexpected_error` | 未預期錯誤中斷了 fusion |

> 📌 最後一個 `fusion_invocation_capped` 與第 14 篇的「遞迴保護」直接相關——Fusion 被設計成**單層**，
> 不允許 panel/judge 再遞迴呼叫自己。

## 4. Web 工具貫穿 panel 與 judge

`openrouter:web_search` 與 `openrouter:web_fetch` 在 **panel 與 judge 兩端**都啟用
（[server-tools 文件](https://openrouter.ai/docs/guides/features/server-tools/fusion)）：

> `openrouter:web_search` and `openrouter:web_fetch` are enabled on both the panel and the judge calls,
> so models can pull fresh sources while they answer and analyze.

這表示：
- panel 模型可以邊回答邊查網路（不是閉卷作答）。
- judge 比對時也能查證，而非只憑 panel 文字。

`max_tool_calls`（預設 8，範圍 1–16）限制每個 panel 模型與 judge 在 web 工具迴圈內最多幾步才須產出文字
（[fusion-router 文件](https://openrouter.ai/docs/guides/routing/routers/fusion-router) 參數表）。
這防止「推理重的模型耗盡 token 預算才開始寫可見文字」（`max_completion_tokens` 的用途，同表）。

## 5. 與 OpenMythos 的對照（為第 17 篇鋪墊）

把 judge 的結構化分析放回 Part I 的視角：

| Fusion judge | OpenMythos LTI 注入 |
|--------------|---------------------|
| 比對多個 panel 的**文字回應** | 注入單一凍結的 `e` 到每個迴圈 |
| 產出**符號化**結構（JSON 五欄位） | 產出**連續**隱藏狀態更新 |
| 一次性（judge 跑一次） | 反覆性（每迴圈都注入） |
| 處理「不同意見」 | 處理「隱藏狀態漂移」 |
| 降級：judge 掛了給原始回應 | 降級：無（但 ρ(A)<1 保證不發散） |

兩者都在做「把多個來源的資訊收斂」，但 Fusion 處理的是**模型間的異質意見**，
Mythos 處理的是**同一模型跨時間的狀態演化**。第 17 篇會把這個對照展開成「整合 vs 注入」的設計映射。

## 6. 一個重要的工程啟示：結構化優於自由文字

Fusion 選擇讓 judge 產出**結構化 JSON**而非「一段綜合文字」，是個可遷移的設計原則：

- 結構化讓下游（outer 模型或程式）能**程式化地**取用共識、矛盾、盲區。
- 它把「事實」（responses）與「判斷」（analysis）分層，便於除錯與審計。
- 它讓降級變得自然：judge 掛了就退回原始 responses，因為兩者本就分開。

這對第 21 篇的設計文件是直接養分：若要打造自己的「整合層」，讓它產出結構化、可比對、可降級的輸出，
遠勝於讓它吐一段自由文字。

## 7. 本文結論與下篇預告

- judge 產出五欄結構化分析：consensus／contradictions／partial_coverage／unique_insights／blind_spots。
- 「compares not merges」是核心：分類、標出處、保留分歧，而非平均。
- 降級分三級：部分 panel 失敗（仍 ok+failed_models）、judge 失敗（ok 但省略 analysis）、全失敗（error+failure_reason）。
- web 工具貫穿 panel 與 judge，受 `max_tool_calls`／`max_completion_tokens` 節流。
- 結構化輸出是可遷移的設計原則，直接餵養第 21 篇。

**下一篇（14）** 談 Fusion 的參數細節、preset 體系，以及最重要的**遞迴保護**（recursion guard）——
為什麼 Fusion 被設計成「單層」、panel/judge 不能再呼叫自己。這與 Mythos 的「可無限加迴圈」形成尖銳對比。

## 引用證據（OpenRouter 官方文件，已 webfetch 確認 HTTP 200）

- 結構化分析五欄位：https://openrouter.ai/docs/guides/features/server-tools/fusion
- 「compares rather than merges」：https://openrouter.ai/docs/guides/routing/routers/fusion-router
- 降級三級（failed_models / judge degradation / hard failures）：https://openrouter.ai/docs/guides/features/server-tools/fusion
- web 工具貫穿：https://openrouter.ai/docs/guides/features/server-tools/fusion
- 參數 max_tool_calls / max_completion_tokens：https://openrouter.ai/docs/guides/routing/routers/fusion-router
