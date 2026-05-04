# AFMIP — Person 4 Power BI Handoff

This file is for the Power BI owner.

## Official Signal File

Use the latest signal export from Azure Blob Storage:

```text
Storage account: afmipdevdp9fba
Container: afmip-exports
Blob path: signals/signals_latest.csv
```

Historical copies are stored here:

```text
Container: afmip-exports
Blob path: signals/history/
```

Current historical file:

```text
signals/history/signals_20260504_0853.csv
```

## Supported Horizons

Build visuals only for:

- `Next day`
- `1 week`

Do not build official visuals for:

- `1 month`
- `3 months`
- `1 year`

Those longer horizons are not production-ready with the current dataset.

## CSV Columns

Expected columns:

| Column | Meaning |
|---|---|
| `ticker` | Stock symbol, for example `AAPL` |
| `date` | Market row date used for the prediction |
| `exported_at` | UTC export timestamp |
| `signal_Next_day` | `UP`, `DOWN`, or `HOLD` |
| `confidence_Next_day` | Model probability-like score from `0.0` to `1.0` |
| `signal_1_week` | `UP`, `DOWN`, or `HOLD` |
| `confidence_1_week` | Model probability-like score from `0.0` to `1.0` |

Signal meanings:

- `UP`: model expects upward movement for that horizon
- `DOWN`: model expects downward movement for that horizon
- `HOLD`: model is uncertain; treat as no action

## Load In Power BI

Use one of these options.

### Option A — Azure Blob Storage

1. Open Power BI Desktop.
2. Select `Get Data`.
3. Choose `Azure Blob Storage`.
4. Enter storage account:

```text
afmipdevdp9fba
```

5. Select container:

```text
afmip-exports
```

6. Choose:

```text
signals/signals_latest.csv
```

### Option B — Local CSV

Use this only if Azure access is not available:

```text
azure_runs/affable_bag_pk0m4q6h7x/named-outputs/model_output/exports/signals_20260504_0853.csv
```

## Data Types

Set these types in Power BI:

| Column pattern | Type |
|---|---|
| `ticker` | Text |
| `date` | Date |
| `exported_at` | Date/Time |
| `signal_*` | Text |
| `confidence_*` | Decimal Number |

## Recommended Dashboard

Page 1:

- Card: total tickers
- Card: count of `UP`, `DOWN`, and `HOLD` for `Next day`
- Card: count of `UP`, `DOWN`, and `HOLD` for `1 week`
- Table: `ticker`, `signal_Next_day`, `confidence_Next_day`, `signal_1_week`, `confidence_1_week`
- Bar chart: top 20 tickers by `confidence_Next_day`
- Bar chart: top 20 tickers by `confidence_1_week`

Page 2:

- Ticker slicer
- Signal slicer for `Next day`
- Signal slicer for `1 week`
- Detailed table for selected ticker(s)

## Recommended Filters

For action-focused views:

```text
signal_Next_day <> HOLD
```

For stronger next-day signals:

```text
confidence_Next_day >= 0.52
```

For stronger 1-week signals:

```text
confidence_1_week >= 0.52
```

## Color Rules

Use simple conditional formatting:

| Signal | Color |
|---|---|
| `UP` | Green |
| `DOWN` | Red |
| `HOLD` | Gray |

## Important Notes

- `confidence` is not guaranteed profit.
- `HOLD` should remain visible because it communicates uncertainty.
- The latest completed Azure ML run was:

```text
affable_bag_pk0m4q6h7x
```

- The latest exported Power BI file is:

```text
signals_20260504_0853.csv
```

