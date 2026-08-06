"""
config.py  —  Configuration for Maximum Entropy IRL Engine
===========================================================

Defines:
  - UNIVERSES: ETF ticker sets
  - IRL: Maximum Entropy IRL parameters
  - FEATURES: Feature functions for reward learning
  - WINDOWS: Time windows for reward inference
"""

# ── HuggingFace ──────────────────────────────────────────────────────────────

HF_TOKEN = ""
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-maxent-irl-results"


# ── ETF Universes ────────────────────────────────────────────────────────────

UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}


# ── Windows ──────────────────────────────────────────────────────────────────

WINDOWS = [63, 126, 252, 504]
WINDOW_LABELS = {
    63: "63d  (~3 months) — Short-term",
    126: "126d (~6 months) — Medium-term",
    252: "252d (~1 year) — Core Signal",
    504: "504d (~2 years) — Long-term",
}
PRIMARY_WINDOW = 252


# ── IRL Parameters ──────────────────────────────────────────────────────────

IRL = {
    "learning_rate": 0.01,     # Learning rate for reward optimization
    "n_iterations": 100,       # Maximum iterations
    "convergence_threshold": 1e-4,  # Convergence threshold
    "regularization": 0.01,    # L2 regularization
    "n_trajectories": 50,      # Number of trajectories to sample
}


# ── Feature Functions ──────────────────────────────────────────────────────

FEATURES = {
    "momentum": True,          # Momentum features
    "volatility": True,        # Volatility features
    "skewness": True,          # Skewness features
    "drawdown": True,          # Drawdown features
    "risk_adjusted": True,     # Risk-adjusted returns
    "macro_correlation": True, # Correlation with macro signals
}


# ── Macro Signals ────────────────────────────────────────────────────────────

MACRO_SIGNALS = [
    ("VIX",       "VIX",           0.30, -1.0),
    ("T10Y2Y",    "10Y–2Y Spread", 0.25, +1.0),
    ("DXY",       "DXY",           0.20, -1.0),
    ("IG_SPREAD", "IG Spread",     0.15, -1.0),
    ("HY_SPREAD", "HY Spread",     0.10, -1.0),
]

MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]
