"""
Buffett-style financial analysis pipeline using edgartools.

Implements the requirements from "Python Pipeline Specification for Buffett-Style
Financial Analysis Using edgartools" (provided separately) with multi-year
statement stitching, standardized line-item extraction, metric computation,
qualitative flags, and narrative synthesis.

How to run
----------
1) Install dependencies (recommended: a virtual environment)
   - python -m venv .venv
   - . .venv/bin/activate
   - pip install edgartools pandas numpy matplotlib

2) Set your SEC EDGAR identity (required by SEC)
   - Preferred: export EDGAR_IDENTITY="Name email@domain.com"
   - Or: export EDGAR_EMAIL="email@domain.com"
   - Or: pass --email "Name email@domain.com" on the command line

3) Run from the command line
   - python buffett_pipeline.py AAPL
   - python buffett_pipeline.py 0000320193 --years 10
   - python buffett_pipeline.py AAPL --years 5 --plots-dir plots
   - python buffett_pipeline.py AAPL --no-plots

4) Use from a notebook or another Python module
   - from buffett_pipeline import run_pipeline
   - result = run_pipeline("AAPL", years=10, identity="Name email@domain.com")
   - result.metrics  # DataFrame with periods + mean/std/flag
   - result.narrative  # Narrative summary
   - result.plot_paths  # Plot file paths (if generated)

Outputs
-------
- Prints a metrics table and a narrative summary when run as a script.
- Returns a PipelineResult (metrics DataFrame, narrative string, plot paths, flags)
  when imported and used programmatically.
- Plots are saved to --plots-dir (default: ./plots) unless --no-plots is used.

Notes
-----
- The script attempts to analyze up to --years periods, but will use fewer if
  fewer periods are available in the SEC filings.
- Missing line items are handled gracefully with NaN values and warnings.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from edgar import Company, MultiFinancials, set_identity
from edgar.xbrl.xbrl import XBRLFilingWithNoXbrlData

META_COLUMNS = {
    "concept",
    "label",
    "level",
    "abstract",
    "dimension",
    "member",
    "unit",
    "decimals",
}


@dataclass
class StatementData:
    name: str
    df: pd.DataFrame
    periods: List[str]


@dataclass
class PipelineResult:
    metrics: pd.DataFrame
    narrative: str
    plot_paths: List[str]
    flags: List[str]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _prompt_identity() -> str:
    return input("Enter EDGAR identity (Name email@domain.com): ").strip()


def resolve_identity(identity: Optional[str]) -> str:
    if identity:
        return identity

    env_identity = os.getenv("EDGAR_IDENTITY")
    if env_identity:
        return env_identity

    env_email = os.getenv("EDGAR_EMAIL")
    if env_email:
        return env_email

    return _prompt_identity()


def ensure_identity(identity: Optional[str]) -> str:
    resolved = resolve_identity(identity)
    if not resolved:
        raise ValueError("EDGAR identity is required. Set EDGAR_IDENTITY/EDGAR_EMAIL or provide --email.")
    set_identity(resolved)
    return resolved


def safe_get_financials(company: Company, retries: int = 1, delay: float = 2.0):
    for attempt in range(retries + 1):
        try:
            return company.get_financials()
        except (ConnectionError, TimeoutError, XBRLFilingWithNoXbrlData) as exc:
            logging.warning("Error fetching financials (attempt %s/%s): %s", attempt + 1, retries + 1, exc)
            if attempt < retries:
                time.sleep(delay)
            else:
                raise


def _render_statement(statement) -> pd.DataFrame:
    rendered = statement.render(standard=True)
    try:
        return rendered.to_dataframe(include_dimensions=False)
    except TypeError:
        return rendered.to_dataframe()


def _normalize_statement(statement, n_years: int, name: str) -> Optional[StatementData]:
    if statement is None:
        return None

    df = _render_statement(statement)
    if df is None or df.empty:
        return None

    df = df.copy()

    period_cols = [c for c in df.columns if c not in META_COLUMNS]
    mapped_cols: List[Tuple[pd.Timestamp, str, str]] = []

    for col in period_cols:
        dt = pd.to_datetime(str(col), errors="coerce")
        if pd.isna(dt):
            continue
        label = dt.strftime("%Y-%m-%d")
        mapped_cols.append((dt, label, col))

    mapped_cols.sort(key=lambda x: x[0], reverse=True)

    seen = set()
    rename_map: Dict[str, str] = {}
    ordered_labels: List[str] = []
    for _, label, original in mapped_cols:
        if label in seen:
            continue
        seen.add(label)
        rename_map[original] = label
        ordered_labels.append(label)

    if not ordered_labels:
        return None

    ordered_labels = ordered_labels[:n_years]
    df = df.rename(columns=rename_map)

    keep_cols = ["label"] + [c for c in ordered_labels if c in df.columns]
    df = df[keep_cols]

    for col in ordered_labels:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return StatementData(name=name, df=df, periods=ordered_labels)


def _select_periods(base: StatementData, others: Iterable[StatementData]) -> List[str]:
    base_periods = list(base.periods)
    if not base_periods:
        return []

    period_sets = [set(base_periods)]
    for stmt in others:
        if stmt is None:
            continue
        period_sets.append(set(stmt.periods))

    common = set.intersection(*period_sets) if period_sets else set()
    if common:
        periods = sorted(common, reverse=True)
        if periods != base_periods:
            logging.warning("Period alignment reduced to %s common periods.", len(periods))
        return periods

    logging.warning("No common periods across statements; using income statement periods.")
    return base_periods


def _series_from_row(
    df: pd.DataFrame,
    periods: List[str],
    patterns: List[str],
    positive: bool = False,
) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([np.nan] * len(periods), index=periods)

    labels = df.get("label")
    if labels is None:
        return pd.Series([np.nan] * len(periods), index=periods)

    for pattern in patterns:
        matches = df[labels.str.contains(pattern, case=False, na=False)]
        if not matches.empty:
            values = matches.iloc[0][periods]
            values = pd.to_numeric(values, errors="coerce")
            if positive:
                values = values.abs()
            return values

    logging.warning("Missing row for patterns: %s", patterns)
    return pd.Series([np.nan] * len(periods), index=periods)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def _compute_metrics(
    income: StatementData,
    balance: StatementData,
    cashflow: StatementData,
    periods: List[str],
) -> Tuple[Dict[str, pd.Series], Dict[str, float]]:
    income_df = income.df if income else None
    balance_df = balance.df if balance else None
    cash_df = cashflow.df if cashflow else None

    revenue = _series_from_row(
        income_df,
        periods,
        [r"Total Revenue", r"Net Revenue", r"Sales Revenue", r"Revenue$", r"Net Sales"],
    )
    cogs = _series_from_row(
        income_df,
        periods,
        [r"Cost of Goods Sold", r"Cost of Revenue", r"Cost of Sales"],
        positive=True,
    )
    gross_profit = _series_from_row(
        income_df,
        periods,
        [r"Gross Profit"],
    )
    if gross_profit.isna().all():
        gross_profit = revenue - cogs

    sga = _series_from_row(
        income_df,
        periods,
        [r"Selling.*General.*Administrative", r"SG&A", r"General and Administrative"],
        positive=True,
    )
    rd = _series_from_row(
        income_df,
        periods,
        [r"Research.*Development", r"R&D"],
        positive=True,
    )
    depreciation = _series_from_row(
        income_df,
        periods,
        [r"Depreciation", r"Amortization"],
        positive=True,
    )
    interest_expense = _series_from_row(
        income_df,
        periods,
        [r"Interest Expense", r"Interest and Debt"],
        positive=True,
    )
    operating_income = _series_from_row(
        income_df,
        periods,
        [r"Operating Income", r"Income from Operations", r"Operating Profit"],
    )
    net_income = _series_from_row(
        income_df,
        periods,
        [r"Net Income", r"Net Earnings", r"Profit.*Loss", r"Net Income.*Common"],
    )
    diluted_shares = _series_from_row(
        income_df,
        periods,
        [r"Weighted Average.*Diluted", r"Diluted Shares", r"Shares.*Diluted"],
    )

    total_liabilities = _series_from_row(
        balance_df,
        periods,
        [r"Total Liabilities", r"Liabilities$"],
    )
    equity = _series_from_row(
        balance_df,
        periods,
        [r"Stockholders.*Equity", r"Shareholders.*Equity", r"Total Equity"],
    )

    operating_cash_flow = _series_from_row(
        cash_df,
        periods,
        [r"Net Cash.*Operations", r"Operating.*Cash", r"Cash.*Operating"],
    )
    capex = _series_from_row(
        cash_df,
        periods,
        [r"Capital Expenditures", r"Payments.*Property", r"Purchase.*Property", r"Property.*Plant.*Equipment"],
        positive=True,
    )
    buybacks = _series_from_row(
        cash_df,
        periods,
        [r"Common Stock Repurchased", r"Repurchase.*Stock", r"Treasury Stock", r"Stock Repurchase"],
        positive=True,
    )

    gross_margin = _safe_divide(gross_profit, revenue)
    sga_load = _safe_divide(sga, gross_profit)
    rd_intensity = _safe_divide(rd, revenue.fillna(gross_profit))
    interest_burden = _safe_divide(interest_expense, operating_income)
    debt_to_equity = _safe_divide(total_liabilities, equity)

    equity_next = equity.shift(-1)
    equity_avg = (equity + equity_next) / 2
    equity_avg = equity_avg.where(~equity_avg.isna(), equity)
    roe = _safe_divide(net_income, equity_avg)

    capex_to_earnings = _safe_divide(capex, net_income)
    depreciation_to_capex = _safe_divide(depreciation, capex)

    fcf = operating_cash_flow - capex
    eps = _safe_divide(net_income, diluted_shares)

    metrics = {
        "gross_margin": gross_margin,
        "sga_load": sga_load,
        "rd_intensity": rd_intensity,
        "interest_burden": interest_burden,
        "debt_to_equity": debt_to_equity,
        "return_on_equity": roe,
        "capex_to_net_earnings": capex_to_earnings,
        "depreciation_to_capex": depreciation_to_capex,
        "free_cash_flow": fcf,
        "eps": eps,
        "buybacks": buybacks,
    }

    summary = {
        "total_capex": float(np.nansum(capex.values)),
        "total_net_income": float(np.nansum(net_income.values)),
    }

    return metrics, summary


def _flag_metric(name: str, series: pd.Series) -> str:
    cleaned = series.dropna()
    if cleaned.empty:
        return "missing"

    mean = cleaned.mean()
    std = cleaned.std()

    ratio_metrics = {
        "gross_margin",
        "sga_load",
        "rd_intensity",
        "interest_burden",
        "debt_to_equity",
        "return_on_equity",
        "capex_to_net_earnings",
        "depreciation_to_capex",
    }

    if name in ratio_metrics:
        if mean < 0:
            return "negative"
        if mean > 1.5:
            return "high"
        if std > 0.1:
            return "volatile"

    if name == "free_cash_flow" and (cleaned < 0).sum() > len(cleaned) / 2:
        return "mostly_negative"

    if name == "eps" and (cleaned < 0).any():
        return "negative"

    return ""


def _metric_table(metrics: Dict[str, pd.Series], periods: List[str]) -> pd.DataFrame:
    df = pd.DataFrame({name: series for name, series in metrics.items()}).T
    df = df[periods]
    df["mean"] = df.mean(axis=1, skipna=True)
    df["std"] = df.std(axis=1, skipna=True)
    df["flag"] = [_flag_metric(name, series) for name, series in metrics.items()]
    return df


def _trend_is_declining(series: pd.Series, min_periods: int = 3) -> bool:
    cleaned = series.dropna()
    if len(cleaned) < min_periods:
        return False
    recent = cleaned.iloc[:min_periods]
    return all(recent.values[i] < recent.values[i + 1] for i in range(len(recent.values) - 1))


def _trend_is_rising(series: pd.Series, min_periods: int = 3) -> bool:
    cleaned = series.dropna()
    if len(cleaned) < min_periods:
        return False
    recent = cleaned.iloc[:min_periods]
    return all(recent.values[i] > recent.values[i + 1] for i in range(len(recent.values) - 1))


def _stability_comment(series: pd.Series, label: str) -> Optional[str]:
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    mean = cleaned.mean()
    std = cleaned.std()
    if mean == 0 or np.isnan(mean) or np.isnan(std):
        return None
    if std <= 0.02:
        return f"{label} has stayed within about ±2% of the mean."
    if abs(std / mean) <= 0.15:
        return f"{label} shows relatively stable variation."
    return f"{label} shows noticeable volatility."


def _collect_flags(metrics: Dict[str, pd.Series]) -> List[str]:
    flags: List[str] = []

    if _trend_is_declining(metrics.get("gross_margin", pd.Series(dtype=float))):
        flags.append("Gross margin is declining in the most recent periods.")

    if _trend_is_rising(metrics.get("debt_to_equity", pd.Series(dtype=float))):
        flags.append("Debt-to-equity is rising across recent periods.")

    capex_vs_earnings = metrics.get("capex_to_net_earnings", pd.Series(dtype=float))
    if not capex_vs_earnings.dropna().empty:
        if (capex_vs_earnings > 1).sum() > len(capex_vs_earnings.dropna()) / 2:
            flags.append("Capital expenditures exceed net earnings in most periods.")

    fcf = metrics.get("free_cash_flow", pd.Series(dtype=float))
    if not fcf.dropna().empty:
        if (fcf < 0).sum() > len(fcf.dropna()) / 2:
            flags.append("Free cash flow is negative in most periods.")

    eps = metrics.get("eps", pd.Series(dtype=float))
    buybacks = metrics.get("buybacks", pd.Series(dtype=float))
    if not eps.dropna().empty and not buybacks.dropna().empty:
        if _trend_is_rising(eps) and (buybacks > 0).sum() >= 2:
            flags.append("EPS growth is accompanied by significant buybacks; verify net income growth.")

    return flags


def _narrative(
    company: Company,
    metrics: Dict[str, pd.Series],
    summary: Dict[str, float],
    flags: List[str],
) -> str:
    lines: List[str] = []
    company_name = getattr(company, "name", None) or str(company)

    gross_margin = metrics.get("gross_margin", pd.Series(dtype=float))
    sga_load = metrics.get("sga_load", pd.Series(dtype=float))
    rd_intensity = metrics.get("rd_intensity", pd.Series(dtype=float))
    debt_to_equity = metrics.get("debt_to_equity", pd.Series(dtype=float))
    roe = metrics.get("return_on_equity", pd.Series(dtype=float))
    fcf = metrics.get("free_cash_flow", pd.Series(dtype=float))

    lines.append(f"Buffett-style financial snapshot for {company_name}:")

    gm_comment = _stability_comment(gross_margin, "Gross margin")
    if gm_comment:
        lines.append(gm_comment)

    sga_mean = sga_load.dropna().mean() if not sga_load.dropna().empty else np.nan
    if not np.isnan(sga_mean):
        lines.append(f"SG&A load averages about {sga_mean:.2f} of gross profit.")

    rd_mean = rd_intensity.dropna().mean() if not rd_intensity.dropna().empty else np.nan
    if not np.isnan(rd_mean):
        lines.append(f"R&D intensity averages about {rd_mean:.2f} of revenue.")

    debt_mean = debt_to_equity.dropna().mean() if not debt_to_equity.dropna().empty else np.nan
    if not np.isnan(debt_mean):
        lines.append(f"Debt-to-equity averages around {debt_mean:.2f}.")

    roe_comment = _stability_comment(roe, "ROE")
    if roe_comment:
        lines.append(roe_comment)

    total_capex = summary.get("total_capex", np.nan)
    total_net = summary.get("total_net_income", np.nan)
    if total_net and not np.isnan(total_capex):
        capex_ratio = total_capex / total_net if total_net else np.nan
        if not np.isnan(capex_ratio):
            lines.append(f"Total capex is {capex_ratio:.2f} of total net earnings over the window.")

    if not fcf.dropna().empty:
        fcf_positive = (fcf > 0).sum()
        lines.append(f"Free cash flow is positive in {fcf_positive} out of {len(fcf.dropna())} periods.")

    if flags:
        lines.append("Red flags: " + "; ".join(flags))
    else:
        lines.append("No major red flags detected based on the specified heuristics.")

    return "\n".join(lines)


def _plot_series(series: pd.Series, title: str, ylabel: str, out_path: str) -> Optional[str]:
    cleaned = series.dropna()
    if cleaned.empty:
        return None

    periods_sorted = sorted(cleaned.index)
    values = cleaned.reindex(periods_sorted)

    plt.figure(figsize=(8, 4))
    plt.plot(periods_sorted, values.values, marker="o")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Period")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()
    return out_path


def run_pipeline(
    identifier: str,
    years: int = 10,
    identity: Optional[str] = None,
    plots_dir: str = "plots",
    generate_plots: bool = True,
) -> PipelineResult:
    if not identifier:
        raise ValueError("Identifier must be provided (ticker or CIK).")

    ensure_identity(identity)

    company = Company(identifier)
    logging.info("Processing %s", identifier)

    financials = safe_get_financials(company)
    if financials is None:
        raise RuntimeError("No XBRL financials available for this company.")

    filings = company.get_filings(form="10-K")
    if filings is None or len(filings) == 0:
        logging.warning("No 10-K filings found; falling back to latest financials.")
        income_stmt = financials.income_statement()
        balance_stmt = financials.balance_sheet()
        cash_stmt = financials.cashflow_statement()
    else:
        filings = filings.head(years)
        multi = MultiFinancials.extract(filings)
        income_stmt = multi.income_statement() or financials.income_statement()
        balance_stmt = multi.balance_sheet() or financials.balance_sheet()
        cash_stmt = multi.cashflow_statement() or financials.cashflow_statement()

    income = _normalize_statement(income_stmt, years, "income")
    balance = _normalize_statement(balance_stmt, years, "balance")
    cashflow = _normalize_statement(cash_stmt, years, "cashflow")

    if not income:
        raise RuntimeError("Income statement data unavailable.")

    periods = _select_periods(income, [balance, cashflow])
    if not periods:
        raise RuntimeError("No valid periods available for analysis.")

    if len(periods) < years:
        logging.warning("Only %s periods available (requested %s).", len(periods), years)

    metrics, summary = _compute_metrics(income, balance, cashflow, periods)
    metrics_df = _metric_table(metrics, periods)

    flags = _collect_flags(metrics)
    narrative = _narrative(company, metrics, summary, flags)

    plot_paths: List[str] = []
    if generate_plots:
        os.makedirs(plots_dir, exist_ok=True)
        plot_specs = [
            ("gross_margin", "Gross Margin", "Ratio", "gross_margin.png"),
            ("eps", "Earnings Per Share", "EPS", "eps_trend.png"),
            ("return_on_equity", "Return on Equity", "ROE", "roe_trend.png"),
            ("free_cash_flow", "Free Cash Flow", "FCF", "fcf_trend.png"),
        ]
        for key, title, ylabel, filename in plot_specs:
            series = metrics.get(key)
            if series is None:
                continue
            out_path = os.path.join(plots_dir, filename)
            created = _plot_series(series, title, ylabel, out_path)
            if created:
                plot_paths.append(created)

    return PipelineResult(metrics=metrics_df, narrative=narrative, plot_paths=plot_paths, flags=flags)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Buffett-style financial analysis using edgartools")
    parser.add_argument("identifier", help="Ticker (e.g., AAPL) or CIK (e.g., 0000320193)")
    parser.add_argument("--years", type=int, default=10, help="Number of years to analyze")
    parser.add_argument("--email", dest="identity", help="EDGAR identity (Name email@domain.com)")
    parser.add_argument("--plots-dir", default="plots", help="Directory to save plots")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    args = _parse_args(argv)

    try:
        result = run_pipeline(
            identifier=args.identifier,
            years=args.years,
            identity=args.identity,
            plots_dir=args.plots_dir,
            generate_plots=not args.no_plots,
        )
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        return 1

    print("\n=== Metrics ===")
    print(result.metrics)
    print("\n=== Narrative ===")
    print(result.narrative)

    if result.plot_paths:
        print("\nPlots saved:")
        for path in result.plot_paths:
            print(f"- {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
