from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from poverty_threshold_baseline import (
	OUTPUT_DIR,
	REAL_DIR,
	build_combined_fies_dataframe,
	clean_numeric_series,
	discover_fies_files,
	first_existing_column,
	parse_baseline_threshold,
	standardize_columns,
)


PROJECTION_HORIZON = 5
RISK_FILE = OUTPUT_DIR / "new_poor_risk.csv"
PROJECTION_OUTPUT = OUTPUT_DIR / "new_poor_projection.csv"
PLOT_OUTPUT = OUTPUT_DIR / "projected_poverty_trend.svg"


def normalize_column_name(column: object) -> str:
	value = str(column).strip().lower()
	value = re.sub(r"[^a-z0-9]+", "_", value)
	return re.sub(r"_+", "_", value).strip("_")


def fit_linear_trend(years: pd.Series, values: pd.Series, future_years: list[int]) -> pd.Series:
	frame = pd.DataFrame(
		{
			"year": pd.to_numeric(years, errors="coerce"),
			"value": pd.to_numeric(values, errors="coerce"),
		}
	)
	frame = frame.dropna(subset=["year", "value"])
	frame = frame.groupby("year", as_index=False)["value"].mean().sort_values("year")

	if frame.empty:
		return pd.Series([pd.NA] * len(future_years), index=future_years, dtype="float64")

	if len(frame) == 1:
		constant = float(frame["value"].iloc[0])
		return pd.Series([constant] * len(future_years), index=future_years, dtype="float64")

	x_values = frame["year"].to_numpy(dtype=float)
	y_values = frame["value"].to_numpy(dtype=float)
	slope, intercept = np.polyfit(x_values, y_values, deg=1)
	predictions = intercept + slope * np.asarray(future_years, dtype=float)
	return pd.Series(predictions, index=future_years, dtype="float64")


def load_yearly_threshold_data() -> pd.DataFrame:
	threshold = parse_baseline_threshold(REAL_DIR / "poverty_threshold.csv")
	if threshold.empty:
		return threshold

	threshold = threshold.copy()
	threshold["year"] = pd.to_numeric(threshold["year"], errors="coerce")
	threshold["poverty_threshold_per_capita"] = pd.to_numeric(
		threshold.get("poverty_threshold_per_capita"), errors="coerce"
	)
	threshold = threshold.dropna(subset=["year", "poverty_threshold_per_capita"])
	return threshold.groupby("year", as_index=False)["poverty_threshold_per_capita"].mean()


def load_yearly_fies_data() -> pd.DataFrame:
	fies_files = discover_fies_files(REAL_DIR)
	combined = build_combined_fies_dataframe(fies_files)
	if combined.empty:
		return combined

	combined = combined.copy()
	combined["year"] = pd.to_numeric(combined.get("year"), errors="coerce")
	combined["family_income"] = pd.to_numeric(combined.get("family_income"), errors="coerce")
	combined["family_expenditure"] = pd.to_numeric(combined.get("family_expenditure"), errors="coerce")
	combined = combined.dropna(subset=["year"])

	aggregations: dict[str, str] = {}
	if "family_income" in combined.columns:
		aggregations["family_income"] = "mean"
	if "family_expenditure" in combined.columns:
		aggregations["family_expenditure"] = "mean"

	if not aggregations:
		return combined[["year"]].drop_duplicates().sort_values("year")

	return combined.groupby("year", as_index=False).agg(aggregations)


def load_yearly_risk_data() -> pd.DataFrame:
	if not RISK_FILE.exists():
		return pd.DataFrame(columns=["year", "new_poor_risk"])

	frame = standardize_columns(pd.read_csv(RISK_FILE, dtype=str, keep_default_na=False))
	if "year" not in frame.columns:
		return pd.DataFrame(columns=["year", "new_poor_risk"])

	risk_column = first_existing_column(
		frame,
		[
			"new_poor_flag",
			"new_poor_risk",
			"new_poor_probability",
			"risk",
		],
	)
	if risk_column is None:
		return pd.DataFrame(columns=["year", "new_poor_risk"])

	frame = frame[["year", risk_column]].copy()
	frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
	frame[risk_column] = pd.to_numeric(frame[risk_column], errors="coerce")
	frame = frame.dropna(subset=["year", risk_column])
	return frame.groupby("year", as_index=False)[risk_column].mean().rename(columns={risk_column: "new_poor_risk"})


def build_projection_frame(horizon: int = PROJECTION_HORIZON) -> pd.DataFrame:
	threshold_yearly = load_yearly_threshold_data()
	fies_yearly = load_yearly_fies_data()
	risk_yearly = load_yearly_risk_data()

	if threshold_yearly.empty and fies_yearly.empty:
		return pd.DataFrame(
			columns=[
				"year",
				"projected_poverty_threshold",
				"projected_family_income",
				"projected_family_expenditure",
				"projected_income_gap",
				"projected_expenditure_ratio",
				"projected_new_poor_risk",
			]
		)

	available_years = pd.concat(
		[
			threshold_yearly[["year"]] if not threshold_yearly.empty else pd.DataFrame(columns=["year"]),
			fies_yearly[["year"]] if not fies_yearly.empty else pd.DataFrame(columns=["year"]),
			risk_yearly[["year"]] if not risk_yearly.empty else pd.DataFrame(columns=["year"]),
		],
		ignore_index=True,
	).dropna()

	if available_years.empty:
		return pd.DataFrame(
			columns=[
				"year",
				"projected_poverty_threshold",
				"projected_family_income",
				"projected_family_expenditure",
				"projected_income_gap",
				"projected_expenditure_ratio",
				"projected_new_poor_risk",
			]
		)

	last_year = int(pd.to_numeric(available_years["year"], errors="coerce").max())
	future_years = list(range(last_year + 1, last_year + horizon + 1))

	threshold_prediction = fit_linear_trend(
		threshold_yearly.get("year", pd.Series(dtype=float)),
		threshold_yearly.get("poverty_threshold_per_capita", pd.Series(dtype=float)),
		future_years,
	)
	family_income_prediction = fit_linear_trend(
		fies_yearly.get("year", pd.Series(dtype=float)),
		fies_yearly.get("family_income", pd.Series(dtype=float)),
		future_years,
	)
	family_expenditure_prediction = fit_linear_trend(
		fies_yearly.get("year", pd.Series(dtype=float)),
		fies_yearly.get("family_expenditure", pd.Series(dtype=float)),
		future_years,
	)
	risk_prediction = fit_linear_trend(
		risk_yearly.get("year", pd.Series(dtype=float)),
		risk_yearly.get("new_poor_risk", pd.Series(dtype=float)),
		future_years,
	)

	result = pd.DataFrame(
		{
			"year": future_years,
			"projected_poverty_threshold": threshold_prediction.reindex(future_years).to_numpy(),
			"projected_family_income": family_income_prediction.reindex(future_years).to_numpy(),
			"projected_family_expenditure": family_expenditure_prediction.reindex(future_years).to_numpy(),
			"projected_new_poor_risk": risk_prediction.reindex(future_years).to_numpy(),
		}
	)

	result["projected_income_gap"] = result["projected_family_income"] - result["projected_poverty_threshold"]
	income = result["projected_family_income"].replace({0: pd.NA})
	result["projected_expenditure_ratio"] = result["projected_family_expenditure"] / income
	gap_ratio = result["projected_income_gap"] / result["projected_poverty_threshold"].replace({0: pd.NA})
	risk_from_gap = 1 / (1 + np.exp(gap_ratio - 1))
	result["projected_new_poor_risk"] = (
		0.35 * result["projected_new_poor_risk"].fillna(0)
		+ 0.65 * risk_from_gap
	).clip(lower=0, upper=1)

	return result[
		[
			"year",
			"projected_poverty_threshold",
			"projected_family_income",
			"projected_family_expenditure",
			"projected_income_gap",
			"projected_expenditure_ratio",
			"projected_new_poor_risk",
		]
	].copy()


def save_projection_plot(history: pd.DataFrame, projection: pd.DataFrame) -> None:
	width = 1000
	height = 600
	margin_left = 90
	margin_right = 30
	margin_top = 40
	margin_bottom = 70
	plot_width = width - margin_left - margin_right
	plot_height = height - margin_top - margin_bottom

	series_map: list[tuple[str, pd.DataFrame, str, str]] = []
	if not history.empty:
		if "poverty_threshold_per_capita" in history.columns:
			series_map.append(("Observed Poverty Threshold", history, "year", "poverty_threshold_per_capita"))
		if "family_income" in history.columns:
			series_map.append(("Observed Family Income", history, "year", "family_income"))
	if not projection.empty:
		series_map.append(("Projected Poverty Threshold", projection, "year", "projected_poverty_threshold"))
		series_map.append(("Projected Family Income", projection, "year", "projected_family_income"))

	points: list[tuple[str, list[tuple[float, float]], str]] = []
	all_x: list[float] = []
	all_y: list[float] = []
	for label, frame, x_column, y_column in series_map:
		series = frame[[x_column, y_column]].dropna()
		if series.empty:
			continue
		coords = list(zip(series[x_column].astype(float).tolist(), series[y_column].astype(float).tolist()))
		points.append((label, coords, y_column))
		all_x.extend([x for x, _ in coords])
		all_y.extend([y for _, y in coords])

	if not points:
		PLOT_OUTPUT.write_text(
			"<svg xmlns='http://www.w3.org/2000/svg' width='1000' height='600'></svg>",
			encoding="utf-8",
		)
		return

	x_min = min(all_x)
	x_max = max(all_x)
	y_min = min(all_y)
	y_max = max(all_y)
	if x_min == x_max:
		x_min -= 1
		x_max += 1
	if y_min == y_max:
		y_min -= 1
		y_max += 1

	def scale_x(value: float) -> float:
		return margin_left + (value - x_min) / (x_max - x_min) * plot_width

	def scale_y(value: float) -> float:
		return margin_top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

	palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
	series_elements: list[str] = []
	legend_elements: list[str] = []
	legend_y = 70

	for index, (label, coords, _) in enumerate(points):
		color = palette[index % len(palette)]
		polyline = " ".join(f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in coords)
		series_elements.append(
			f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{polyline}' />"
		)
		for x, y in coords:
			series_elements.append(f"<circle cx='{scale_x(x):.2f}' cy='{scale_y(y):.2f}' r='4' fill='{color}' />")
		legend_elements.append(
			f"<rect x='{margin_left}' y='{legend_y - 12}' width='14' height='14' fill='{color}' />"
			f"<text x='{margin_left + 22}' y='{legend_y}' font-size='14' fill='#1f2937'>{label}</text>"
		)
		legend_y += 22

	ticks_x = np.linspace(x_min, x_max, num=min(6, max(2, len(sorted(set(all_x))))) )
	ticks_y = np.linspace(y_min, y_max, num=6)
	x_axis = [
		f"<line x1='{margin_left}' y1='{margin_top + plot_height}' x2='{margin_left + plot_width}' y2='{margin_top + plot_height}' stroke='#374151' stroke-width='1.5' />",
		f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + plot_height}' stroke='#374151' stroke-width='1.5' />",
	]

	for tick in ticks_x:
		x = scale_x(float(tick))
		x_axis.append(f"<line x1='{x}' y1='{margin_top + plot_height}' x2='{x}' y2='{margin_top + plot_height + 6}' stroke='#374151' />")
		x_axis.append(f"<text x='{x}' y='{margin_top + plot_height + 24}' font-size='12' text-anchor='middle' fill='#374151'>{int(round(tick))}</text>")

	for tick in ticks_y:
		y = scale_y(float(tick))
		x_axis.append(f"<line x1='{margin_left - 6}' y1='{y}' x2='{margin_left}' y2='{y}' stroke='#374151' />")
		x_axis.append(f"<text x='{margin_left - 10}' y='{y + 4}' font-size='12' text-anchor='end' fill='#374151'>{int(round(tick))}</text>")

	svg_content = [
		f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
		"<rect width='100%' height='100%' fill='white' />",
		"<text x='90' y='28' font-size='20' font-weight='700' fill='#111827'>Poverty Threshold vs Family Income Projection</text>",
		*series_elements,
		*legend_elements,
		*x_axis,
		f"<text x='{margin_left + plot_width / 2}' y='{height - 18}' font-size='13' text-anchor='middle' fill='#374151'>Year</text>",
		f"<text x='20' y='{margin_top + plot_height / 2}' font-size='13' text-anchor='middle' fill='#374151' transform='rotate(-90 20 {margin_top + plot_height / 2})'>Value</text>",
		"</svg>",
	]
	PLOT_OUTPUT.write_text("\n".join(svg_content), encoding="utf-8")


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	threshold_history = load_yearly_threshold_data()
	fies_history = load_yearly_fies_data()
	projection = build_projection_frame()

	projection.to_csv(PROJECTION_OUTPUT, index=False)
	save_projection_plot(threshold_history.merge(fies_history, on="year", how="outer").sort_values("year"), projection)

	print(f"Projected years: {projection['year'].tolist()}")
	print(f"Saved output to: {PROJECTION_OUTPUT}")
	print(f"Saved plot to: {PLOT_OUTPUT}")


if __name__ == "__main__":
	main()
