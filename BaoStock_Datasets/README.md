# BaoStock_Datasets 目录说明

更新时间：2026-07-26

`BaoStock_Datasets/` 是本项目的本地行情数据区。后续训练、评估、数据锁定和数据报备都应优先以这里的数据为准。

该目录只承担数据存放和数据拉取脚本管理，不放模型权重，不放提交包。

## 目录职责

```text
BaoStock_Datasets/
  data/       # 本地行情数据文件
  scripts/    # 数据拉取、数据扩展、早期推理控制脚本
```

## data 子目录

当前主要数据目录：

- `data/full_hfq_baostock_20260703/`
  当前优先使用的数据目录。用于 BaoStock 后复权行情扩展和最新评估。
- `data/full_hfq_baostock_20260703_attempt/`
  BaoStock 拉取过程中的尝试/中间目录，默认不作为正式训练数据源。
- `data/full_hfq_tencent/`
  历史遗留命名目录。后续若要求严格只使用 BaoStock，需要先核验其中每个文件来源，再决定是否继续使用。
- `data/smoke5_baostock/`
  小规模 BaoStock smoke test 数据，用于快速验证数据读取流程。
- `data/smoke5_baostock_script/`
  smoke test 相关脚本或中间产物目录。
- `data/fusion_research_20260726/`
  F/G 系列研究专用冻结数据，包含行业静态映射、中金所 IF/IH/IC/IM 日行情与
  点时财报指标。每类数据都有原始响应/压缩包、处理文件、`validation.json`
  和 `SOURCE_MANIFEST.json`。这些数据尚未与 2026-07-18 赛务报备邮件逐项
  核对，只能用于明确标记的 `TEST` 研究。
- `data/prospective_d_stability_20260727/`
  Candidate-D 8+4 窗口前瞻检验的因果行情前缀。每轮只能在决策日收盘后查询
  至该决策日，禁止未来区间、历史覆盖和模型产物。

## scripts 子目录

当前脚本：

- `download_baostock_expanded_data.py`
  BaoStock 数据拉取/扩展脚本，后续新增数据应优先使用这个方向。
- `download_akshare_expanded_data.py`
  历史遗留脚本。当前项目数据源要求优先 BaoStock，默认不要用于新增数据。
- `rerank_existing_model.py`
  早期基于已有模型结果做 rerank 的辅助脚本。
- `search_inference_controls.py`
  早期搜索推理控制参数的辅助脚本。
- `download_csi300_industry_snapshot.py`
  冻结东方财富公开沪深300成分与行业静态快照；不把当前价格字段写入训练文件。
- `download_cffex_daily_futures.py`
  下载并校验中金所公开月度日行情 ZIP，构造前一交易日持仓量选择的主力合约
  与股指期货市场特征。
- `download_eastmoney_financial_pit.py`
  冻结东方财富主要财务指标，并以
  `max(NOTICE_DATE, UPDATE_DATE)` 作为保守可得日，防止修订值回填。
- `build_prospective_causal_prefix.py`
  为 Candidate-D 前瞻窗口仅拉取上一冻结前缀之后、当前决策日及之前的
  BaoStock 后复权增量，并生成带 SHA-256/MD5 来源清单的训练前缀。
- `build_prospective_outcome_bundle.py`
  只在窗口卖出日 09:31 后拉取该窗口买入日至卖出日的固定股池行情与同期
  上证指数开盘价，并将带边界校验和哈希的 outcome bundle 写入工作区外
  sealed vault；禁止写回训练前缀或 candidate 包。

## 使用边界

- 正式训练数据和评估数据应明确记录文件路径、时间窗口、来源和 md5。
- 不要把模型训练输出写入该目录。
- 不要在未记录来源的情况下混入其他行情源。
- 如果未来需要锁定参赛报备数据，应在这里建立清晰的锁定版本目录和 md5 清单。
