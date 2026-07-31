# Volatility Forecasting Based on DQN

## Prepare intraday returns

Place the raw Chinese A-share quote files in `data/raw/chinese/`, then run:

```bash
python code/scripts/prepare_returns.py
```

The five-minute log-return files are written to
`data/processed/chinese_return/`.
