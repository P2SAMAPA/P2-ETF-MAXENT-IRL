# P2-MAXENT-IRL

**Maximum Entropy Inverse Reinforcement Learning — Inferring Trading Goals from Market Data**

Part of the **P2Quant Engine Suite** · P2SAMAPA

---

## What This Engine Does

This engine uses **Maximum Entropy Inverse Reinforcement Learning** to infer the underlying reward function (trading objectives) from observed market behavior. Instead of specifying trading rules, the engine learns what goals market participants are optimizing for.

### Theory

**Inverse Reinforcement Learning (IRL):**
- Instead of learning a policy from rewards, learn the reward function from observed behavior
- The "expert" (market) is assumed to be optimal
- We infer what the expert is optimizing for

**Maximum Entropy IRL:**
- Models behavior as a softmax over rewards
- Finds the reward function that makes observed behavior most likely
- Handles suboptimal demonstrations naturally

**Feature-Based Reward:**
- Reward is a linear combination of features
- Features include momentum, volatility, drawdown, etc.
- IRL learns the weights for each feature

---

## Key Metrics

| Metric | What it tells you |
|--------|-------------------|
| **z-score** | Cross-sectional ranking of inferred reward |
| **Reward** | Inferred value of the asset |
| **Weights** | Feature importance in the reward function |
| **Converged** | Whether IRL converged successfully |

---

## Features

| Feature | What it captures |
|---------|------------------|
| **Momentum** | Trend-following behavior |
| **Volatility** | Risk aversion/preference |
| **Skewness** | Preference for asymmetry |
| **Drawdown** | Loss aversion |
| **Risk-Adjusted** | Sharpe ratio preferences |
| **Macro Correlation** | Macro sensitivity |

---

## Windows

| Window | Purpose |
|--------|---------|
| 63d | Short-term reward inference |
| 126d | Medium-term reward inference |
| 252d | Core signal (primary) |
| 504d | Long-term reward inference |

---

## Universes

| Universe | Tickers |
|----------|---------|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

---

## Interpretation

| z-score | Action | Meaning |
|---------|--------|---------|
| **> 0.5** | BUY | ETF has high inferred reward |
| **-0.5 to 0.5** | HOLD | Neutral inferred reward |
| **< -0.5** | SELL | ETF has low inferred reward |

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-MAXENT-IRL
cd P2-MAXENT-IRL
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py

streamlit run streamlit_app.py
GitHub Actions
Runs automatically at 00:30 UTC Monday–Saturday via .github/workflows/daily.yml.

Required secret: HF_TOKEN

References
Ziebart, B. D., et al. (2008). Maximum Entropy Inverse Reinforcement Learning. AAAI.

Ng, A. Y., & Russell, S. J. (2000). Algorithms for Inverse Reinforcement Learning. ICML.

Levine, S., & Abbeel, P. (2014). Learning Neural Network Policies with Guided Policy Search. ICML.
