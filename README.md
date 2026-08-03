# Volatility Forecasting Based on DQN

## Prepare intraday returns

Place the raw Chinese A-share quote files in `data/raw/chinese/`, then run:

```bash
python code/scripts/prepare_returns.py
```

The five-minute log-return files are written to
`data/processed/chinese_return/`.

## Train rolling linear candidates

Train the pooled OLS and HAR models separately for daily and hourly RV:

```bash
python code/scripts/train_linear_candidates.py --frequency 1D
python code/scripts/train_linear_candidates.py --frequency 1H
```

Checkpoints are written to `checkpoints/candidates/`, while out-of-sample
forecasts and MSE/QLIKE summaries are written to `results/`.

## Train rolling neural candidates

Install the PyTorch dependency, then train the pooled LSTM and three TCN
candidates:

```bash
uv sync
python code/scripts/train_neural_candidates.py --frequency 1D
python code/scripts/train_neural_candidates.py --frequency 1H
```

Each four-year window uses its final six months for neural-network validation
and early stopping. Neural checkpoints are written as `.pt` files, and their
out-of-sample forecasts and metrics are written to `results/`.
