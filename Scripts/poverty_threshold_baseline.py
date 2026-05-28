from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REAL_DIR = BASE_DIR / "data" / "real"
OUTPUT_DIR = BASE_DIR / "outputs"
BASELINE_FILE = REAL_DIR / "poverty_threshold.csv"
FIES_KEYWORDS = ("income", "expenditure", "total", "family", "fies", "per_capita", "decile", "threshold")


def normalize_column_name(column: object) -> str:
	value = str(column).strip().lower()
	value = re.sub(r"[^a-z0-9]+", "_", value)
	return re.sub(r"_+", "_", value).strip("_")


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
	standardized = frame.copy()
	standardized.columns = [normalize_column_name(column) for column in standardized.columns]
	return standardized


def clean_area_name(value: object) -> str:
	cleaned = str(value).strip().strip('"')
	cleaned = re.sub(r"(\d+/?)$", "", cleaned)
	cleaned = re.sub(r"\s+\d+/?$", "", cleaned)
	cleaned = re.sub(r"\s+", " ", cleaned)
	return cleaned.strip()


def clean_numeric_series(series: pd.Series) -> pd.Series:
	cleaned = series.astype(str).str.strip()
	cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
	cleaned = cleaned.str.replace(r"[%,$]", "", regex=True)
	cleaned = cleaned.str.replace(r"[^0-9.\-]", "", regex=True)
	cleaned = cleaned.replace({"": pd.NA, "-": pd.NA, ".": pd.NA})
	return pd.to_numeric(cleaned, errors="coerce")


def first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
	for candidate in candidates:
		if candidate in frame.columns:
			return candidate
	return None


def first_valid_value(series: pd.Series):
	non_null = series.dropna()
	if non_null.empty:
		return pd.NA
	return non_null.iloc[0]


def join_unique_values(series: pd.Series) -> str | None:
	values = [str(value) for value in series.dropna().astype(str).tolist() if str(value).strip()]
	unique_values = list(dict.fromkeys(values))
	if not unique_values:
		return pd.NA
	return ", ".join(unique_values)


def extract_year_from_name(path: Path) -> int | None:
	match = re.search(r"(20\d{2})", path.stem)
	if match:
		return int(match.group(1))
	return pd.NA


def discover_fies_files(real_dir: Path) -> list[Path]:
	csv_files = sorted(path for path in real_dir.glob("*.csv") if path.name.lower() != BASELINE_FILE.name.lower())
	matched = [path for path in csv_files if any(keyword in path.name.lower() for keyword in FIES_KEYWORDS)]
	return matched if matched else csv_files


def parse_baseline_threshold(path: Path) -> pd.DataFrame:
	frame = standardize_columns(pd.read_csv(path, dtype=str, keep_default_na=False))
	year_column = first_existing_column(frame, ["year"])
	threshold_column = first_existing_column(
		frame,
		[
			"poverty_threshold_per_capita",
			"poverty_threshold",
			"threshold_per_capita",
			"poverty_line",
			"poverty_threshold_per_capita_income",
		],
	)
	area_column = first_existing_column(frame, ["area", "region", "province", "huc", "location"])

	result = pd.DataFrame()
	if year_column is not None:
		result["year"] = pd.to_numeric(frame[year_column], errors="coerce")
	if area_column is not None:
		result["area"] = frame[area_column].replace({"": pd.NA})
	if threshold_column is not None:
		result["poverty_threshold_per_capita"] = clean_numeric_series(frame[threshold_column])

	if "source_file" in frame.columns:
		result["source_file"] = frame["source_file"].replace({"": pd.NA})
	else:
		result["source_file"] = path.name

	return result.dropna(subset=["year"]).copy()


def parse_wide_table(path: Path, value_column_name: str, year_positions: dict[int, int], multiplier: float = 1.0) -> pd.DataFrame:
	raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
	records: list[dict[str, object]] = []

	for _, row in raw.iterrows():
		area = clean_area_name(row.iloc[0]) if len(row) else ""
		if not area:
			continue
		if area.lower() in {"region/province/huc", "region/province/hucs", "region/province"}:
			continue
		if area.startswith("(") or area.lower().startswith("percent change"):
			continue

		values: dict[int, object] = {}
		for year, position in year_positions.items():
			if position < len(row):
				numeric_value = clean_numeric_series(pd.Series([row.iloc[position]])).iloc[0]
				if pd.notna(numeric_value):
					numeric_value = float(numeric_value) * multiplier
				values[year] = numeric_value
			else:
				values[year] = pd.NA

		if all(pd.isna(value) for value in values.values()):
			continue

		for year, value in values.items():
			if pd.notna(value):
				records.append(
					{
						"year": year,
						"area": area,
						value_column_name: value,
						"source_file": path.name,
					}
				)

	return pd.DataFrame.from_records(records)


def parse_row_level_file(path: Path) -> pd.DataFrame:
	frame = standardize_columns(pd.read_csv(path, dtype=str, keep_default_na=False))
	result = pd.DataFrame()

	area_column = first_existing_column(frame, ["region", "province", "huc", "area", "location"])
	if area_column is not None:
		result["area"] = frame[area_column].map(lambda value: clean_area_name(value) if str(value).strip() else pd.NA)

	income_column = first_existing_column(
		frame,
		[
			"total_household_income",
			"total_income",
			"family_income",
			"annual_family_income",
			"income",
		],
	)
	expenditure_column = first_existing_column(
		frame,
		[
			"total_food_expenditure",
			"family_expenditure",
			"total_expenditure",
			"annual_family_expenditure",
			"expenditure",
		],
	)

	if income_column is not None:
		result["family_income_raw"] = clean_numeric_series(frame[income_column])
	if expenditure_column is not None:
		result["family_expenditure_raw"] = clean_numeric_series(frame[expenditure_column])

	year_value = extract_year_from_name(path)
	if pd.notna(year_value):
		result["year"] = year_value

	result["source_file"] = path.name
	return result.dropna(subset=["area"], how="all").copy()


def parse_fies_file(path: Path) -> pd.DataFrame:
	lowered_name = path.name.lower()
	if "poverty_threshold" in lowered_name:
		return pd.DataFrame()
	if "table 1a" in lowered_name or "cv_" in lowered_name or "coefficient" in lowered_name:
		return pd.DataFrame()

	if "table 1" in lowered_name:
		return parse_wide_table(path, "family_income", {2018: 1, 2021: 12, 2023: 23}, multiplier=1000.0)
	if "table 2" in lowered_name:
		return parse_wide_table(path, "family_expenditure", {2018: 1, 2021: 12, 2023: 23}, multiplier=1000.0)
	if "table 3" in lowered_name:
		return parse_wide_table(path, "family_income_total", {2018: 1, 2021: 12, 2023: 23}, multiplier=1_000_000.0)
	return parse_row_level_file(path)


def build_combined_fies_dataframe(paths: list[Path]) -> pd.DataFrame:
	frames: list[pd.DataFrame] = []
	source_priority = {
		"family_income": 0,
		"family_expenditure": 0,
		"family_income_total": 1,
		"family_income_raw": 2,
		"family_expenditure_raw": 2,
	}

	for path in paths:
		frame = parse_fies_file(path)
		if frame.empty:
			continue
		frame = frame.copy()
		frame["source_priority"] = 3
		for column in source_priority:
			if column in frame.columns:
				frame.loc[frame[column].notna(), "source_priority"] = source_priority[column]
		frames.append(frame)

	if not frames:
		return pd.DataFrame()

	combined = pd.concat(frames, ignore_index=True, sort=False)
	if "year" in combined.columns:
		combined["year"] = pd.to_numeric(combined["year"], errors="coerce")
	if "area" in combined.columns:
		combined["area"] = combined["area"].replace({"": pd.NA})

	sort_columns = [column for column in ["year", "area", "source_priority"] if column in combined.columns]
	if sort_columns:
		combined = combined.sort_values(sort_columns, na_position="last")

	aggregation: dict[str, object] = {}
	for column in [
		"family_income",
		"family_expenditure",
		"family_income_total",
		"family_income_raw",
		"family_expenditure_raw",
	]:
		if column in combined.columns:
			aggregation[column] = first_valid_value
	if "source_file" in combined.columns:
		aggregation["source_file"] = join_unique_values

	group_keys = [column for column in ["year", "area"] if column in combined.columns]
	if not group_keys:
		return combined

	collapsed = combined.groupby(group_keys, dropna=False, as_index=False).agg(aggregation)

	if "family_income" not in collapsed.columns and "family_income_total" in collapsed.columns:
		collapsed["family_income"] = collapsed["family_income_total"]
	if "family_income" not in collapsed.columns and "family_income_raw" in collapsed.columns:
		collapsed["family_income"] = collapsed["family_income_raw"]
	if "family_expenditure" not in collapsed.columns and "family_expenditure_raw" in collapsed.columns:
		collapsed["family_expenditure"] = collapsed["family_expenditure_raw"]

	return collapsed


def load_and_merge_data() -> pd.DataFrame:
	baseline = parse_baseline_threshold(BASELINE_FILE)
	fies_files = discover_fies_files(REAL_DIR)
	combined_fies = build_combined_fies_dataframe(fies_files)

	if combined_fies.empty:
		return combined_fies

	if "year" in combined_fies.columns:
		combined_fies = combined_fies.dropna(subset=["year"])

	join_keys = ["year"] if "year" in baseline.columns and "year" in combined_fies.columns else []
	if "area" in baseline.columns and "area" in combined_fies.columns:
		baseline_areas = baseline["area"].dropna().astype(str).str.strip()
		fies_areas = combined_fies["area"].dropna().astype(str).str.strip()
		if not baseline_areas.empty and not fies_areas.empty:
			join_keys = ["year", "area"] if join_keys else ["area"]

	merged = combined_fies.merge(baseline, on=join_keys, how="left", suffixes=("", "_baseline"))

	if "poverty_threshold_per_capita" not in merged.columns:
		threshold_column = first_existing_column(
			merged,
			[
				"poverty_threshold_per_capita",
				"poverty_threshold",
				"threshold_per_capita",
				"poverty_line",
			],
		)
		if threshold_column is not None:
			merged = merged.rename(columns={threshold_column: "poverty_threshold_per_capita"})

	if "family_income" not in merged.columns and "family_income_total" in merged.columns:
		merged["family_income"] = merged["family_income_total"]
	if "family_income" not in merged.columns and "family_income_raw" in merged.columns:
		merged["family_income"] = merged["family_income_raw"]
	if "family_expenditure" not in merged.columns and "family_expenditure_raw" in merged.columns:
		merged["family_expenditure"] = merged["family_expenditure_raw"]

	merged["poverty_threshold_per_capita"] = pd.to_numeric(merged.get("poverty_threshold_per_capita"), errors="coerce")
	merged["family_income"] = pd.to_numeric(merged.get("family_income"), errors="coerce")
	merged["family_expenditure"] = pd.to_numeric(merged.get("family_expenditure"), errors="coerce")

	merged["poverty_gap"] = merged["family_income"] - merged["poverty_threshold_per_capita"]
	merged["expenditure_ratio"] = merged["family_expenditure"] / merged["family_income"].where(merged["family_income"] != 0)
	merged["new_poor_flag"] = pd.Series(pd.NA, index=merged.index, dtype="object")
	merged.loc[merged["poverty_gap"].notna(), "new_poor_flag"] = (merged.loc[merged["poverty_gap"].notna(), "poverty_gap"] <= 0).astype(int)

	output_columns = [
		column
		for column in [
			"year",
			"area",
			"poverty_threshold_per_capita",
			"family_income",
			"family_expenditure",
			"poverty_gap",
			"expenditure_ratio",
			"new_poor_flag",
			"source_file",
		]
		if column in merged.columns
	]
	return merged[output_columns].copy()


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	result = load_and_merge_data()
	output_path = OUTPUT_DIR / "new_poor_risk.csv"
	result.to_csv(output_path, index=False)
	print(f"Loaded {len(result)} rows")
	print(str(output_path))


if __name__ == "__main__":
	main()
