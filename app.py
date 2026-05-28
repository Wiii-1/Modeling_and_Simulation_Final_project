from pathlib import Path
import csv
import re

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
RISK_PATH = BASE_DIR / "outputs" / "new_poor_risk.csv"
PROJECTION_PATH = BASE_DIR / "outputs" / "new_poor_projection.csv"
CITY_POP_PATH = BASE_DIR / "data" / "real" / "pasay-city-population-2026.csv"
PC_WAM_PATH = BASE_DIR / "data" / "real" / "PC WAM 2024-Year-Age.csv"


def clean_text_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\u202f", " ").replace("\u2009", " ").replace("\xa0", " ")
    return text.strip()


def normalize_column_name(name: str) -> str:
    normalized = clean_text_value(name).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def parse_numeric_value(value):
    text = clean_text_value(value)
    if not text or text in {"-", "–", "—"}:
        return pd.NA
    text = text.replace(",", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text or text in {"-", ".", "-."}:
        return pd.NA
    try:
        numeric = float(text)
    except Exception:
        return pd.NA
    if numeric.is_integer():
        return int(numeric)
    return numeric


def parse_year(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"(19|20)\d{2}", str(value))
    if match:
        return int(match.group(0))
    return pd.NA


def safe_load_csv(
    path: Path,
    rename_map: dict[str, list[str]] | None = None,
    text_columns: set[str] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    frame = frame.rename(columns={column: normalize_column_name(column) for column in frame.columns})

    aliases = {
        "year": ["year", "period", "fiscal_year"],
        "area": ["area", "region", "province", "huc"],
        "city": ["city"],
        "population": ["population", "pop2026", "latest_population", "city_population"],
        "poverty_threshold_per_capita": [
            "poverty_threshold_per_capita",
            "poverty_threshold",
            "threshold",
            "poverty_line",
            "projected_poverty_threshold",
        ],
        "family_income": [
            "family_income",
            "average_family_income",
            "mean_family_income",
            "mean_per_capita_income",
            "income",
            "average_income",
            "projected_family_income",
        ],
        "family_expenditure": [
            "family_expenditure",
            "average_family_expenditure",
            "mean_family_expenditure",
            "expenditure",
            "expense",
            "projected_family_expenditure",
        ],
        "poverty_gap": ["poverty_gap", "income_gap", "threshold_gap", "gap", "projected_income_gap"],
        "new_poor_flag": ["new_poor_flag", "new_poor_count", "new_poor_risk", "newly_poor", "new_poor_projection", "projected_new_poor_risk"],
        "expenditure_ratio": ["expenditure_ratio", "projected_expenditure_ratio"],
        "source_file": ["source_file", "source"],
    }
    if rename_map:
        aliases.update(rename_map)

    current = set(frame.columns)
    for canonical, names in aliases.items():
        if canonical in current:
            continue
        for alias in names:
            alias_norm = normalize_column_name(alias)
            if alias_norm in current:
                frame = frame.rename(columns={alias_norm: canonical})
                current = set(frame.columns)
                break

    if "year" in frame.columns:
        frame["year"] = frame["year"].apply(parse_year)
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")

    text_columns = set(text_columns or set()) | {"area", "source_file", "city", "barangay", "age_group", "sex", "label"}
    for column in frame.columns:
        if column not in text_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def extract_latest_year(frame: pd.DataFrame):
    if frame.empty or "year" not in frame.columns:
        return None
    years = frame["year"].dropna()
    if years.empty:
        return None
    return int(years.max())


def latest_year_frame(frame: pd.DataFrame) -> pd.DataFrame:
    latest_year = extract_latest_year(frame)
    if latest_year is None:
        return frame.copy()
    return frame.loc[frame["year"] == latest_year].copy()


def latest_year_snapshot(frame: pd.DataFrame) -> pd.Series:
    latest = latest_year_frame(frame)
    if latest.empty:
        return pd.Series(dtype="object")
    snapshot = pd.Series(dtype="object")
    numeric_columns = latest.select_dtypes(include="number").columns
    for column in numeric_columns:
        snapshot[column] = latest[column].mean(skipna=True)
    for column in latest.columns:
        if column in numeric_columns:
            continue
        values = latest[column].dropna()
        if not values.empty:
            snapshot[column] = values.iloc[-1]
    return snapshot


def yearly_aggregate(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty or "year" not in frame.columns:
        return pd.DataFrame()
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.DataFrame()
    data = frame[["year", *available]].copy()
    data = data.loc[data["year"].notna()].copy()
    if data.empty:
        return pd.DataFrame()
    aggregated = data.groupby("year", as_index=False).mean(numeric_only=True)
    return aggregated.sort_values("year").reset_index(drop=True)


def get_snapshot_value(snapshot: pd.Series, column: str):
    if snapshot.empty or column not in snapshot.index:
        return None
    value = snapshot[column]
    return None if pd.isna(value) else value


def format_population(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(round(float(value))):,}"


def format_currency(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"₱{float(value):,.2f}"


def format_percent(value, ratio: bool = False):
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value) * 100 if ratio else float(value)
    return f"{numeric_value:.2f}%"


def extract_pasay_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    candidate_columns = [column for column in ("area", "city") if column in frame.columns]
    if not candidate_columns:
        return pd.DataFrame(columns=frame.columns)
    mask = pd.Series(False, index=frame.index)
    for column in candidate_columns:
        values = frame[column].fillna("").astype(str).str.lower()
        mask |= values.str.contains(r"\bpasay\b", regex=True, na=False)
    if not mask.any():
        return pd.DataFrame(columns=frame.columns)
    return frame.loc[mask].copy()


def extract_latest_city_population(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or "city" not in frame.columns or "population" not in frame.columns:
        return pd.Series(dtype="object")
    pasay_rows = frame.loc[frame["city"].fillna("").astype(str).str.contains(r"pasay", case=False, regex=True, na=False)].copy()
    if pasay_rows.empty:
        return pd.Series(dtype="object")
    pasay_rows = pasay_rows.sort_values("population", ascending=False)
    return pasay_rows.iloc[0]


def load_pasay_census_context(path: Path) -> dict[str, object]:
    empty_frame = pd.DataFrame(columns=["label", "total_population", "male_population", "female_population"])
    result: dict[str, object] = {
        "city_total_population": pd.NA,
        "city_male_population": pd.NA,
        "city_female_population": pd.NA,
        "age_groups": empty_frame,
        "barangay_totals": pd.DataFrame(columns=["barangay", "total_population", "male_population", "female_population"]),
        "barangay_count": 0,
    }
    if not path.exists():
        return result

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
    except Exception:
        return result

    city_age_rows: list[dict[str, object]] = []
    barangay_rows: list[dict[str, object]] = []
    current_barangay = None
    current_section = None

    for row in rows:
        values = [clean_text_value(cell) for cell in row]
        nonempty = [value for value in values if value]
        if not nonempty:
            continue

        first = nonempty[0]
        first_normalized = normalize_column_name(first)

        if first_normalized.startswith("table_17"):
            continue
        if first_normalized == "sex":
            continue
        if first_normalized == "barangay" and len(nonempty) == 1:
            continue
        if first_normalized == "age_group" and len(nonempty) == 1:
            current_section = "age"
            continue
        if first_normalized.startswith("barangay") and len(nonempty) == 1:
            current_barangay = first
            current_section = "barangay"
            continue

        if first_normalized == "total":
            total_value = parse_numeric_value(nonempty[1] if len(nonempty) > 1 else pd.NA)
            male_value = parse_numeric_value(nonempty[2] if len(nonempty) > 2 else pd.NA)
            female_value = parse_numeric_value(nonempty[3] if len(nonempty) > 3 else pd.NA)
            if current_barangay is None:
                result["city_total_population"] = total_value
                result["city_male_population"] = male_value
                result["city_female_population"] = female_value
            else:
                barangay_rows.append(
                    {
                        "barangay": current_barangay,
                        "total_population": total_value,
                        "male_population": male_value,
                        "female_population": female_value,
                    }
                )
            continue

        if current_section == "age" and current_barangay is None and len(nonempty) >= 4:
            age_label = first
            if age_label.lower() != "total":
                city_age_rows.append(
                    {
                        "label": age_label,
                        "total_population": parse_numeric_value(nonempty[1]),
                        "male_population": parse_numeric_value(nonempty[2]),
                        "female_population": parse_numeric_value(nonempty[3]),
                    }
                )

    age_groups = pd.DataFrame(city_age_rows)
    if not age_groups.empty:
        age_groups = age_groups.loc[age_groups["label"].notna()].copy()
        age_groups = age_groups.loc[age_groups["label"].astype(str).str.lower() != "total"].copy()
        age_groups = age_groups.sort_values("total_population", ascending=False, na_position="last").reset_index(drop=True)

    barangay_totals = pd.DataFrame(barangay_rows)
    if not barangay_totals.empty:
        barangay_totals = barangay_totals.loc[barangay_totals["barangay"].notna()].copy()
        barangay_totals = barangay_totals.sort_values("total_population", ascending=False, na_position="last").reset_index(drop=True)

    result["age_groups"] = age_groups if not age_groups.empty else empty_frame
    result["barangay_totals"] = barangay_totals if not barangay_totals.empty else result["barangay_totals"]
    result["barangay_count"] = int(barangay_totals["barangay"].nunique()) if not barangay_totals.empty else 0
    return result


def build_metric_cards(title: str, snapshot: pd.Series, specs: list[tuple[str, str, str, bool]]) -> None:
    st.subheader(title)
    columns = st.columns(len(specs))
    for column, spec in zip(columns, specs):
        label, metric_key, formatter, is_ratio = spec
        value = get_snapshot_value(snapshot, metric_key)
        if formatter == "currency":
            formatted = format_currency(value)
        elif formatter == "percent":
            formatted = format_percent(value, ratio=is_ratio)
        elif formatter == "population":
            formatted = format_population(value)
        else:
            formatted = "N/A" if value is None else str(value)
        with column:
            st.metric(label, formatted)


def build_dark_figure(title: str, height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=30, r=20, t=60, b=30),
        title=title,
        legend_title_text="",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def add_empty_annotation(fig: go.Figure, message: str) -> go.Figure:
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14),
    )
    return fig


def build_comparison_figure(baseline_frame: pd.DataFrame, projection_frame: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Poverty threshold", "Family income", "Family expenditure", "Income gap"),
    )

    metric_specs = [
        ("poverty_threshold_per_capita", "Baseline poverty threshold", "Projected poverty threshold"),
        ("family_income", "Baseline family income", "Projected family income"),
        ("family_expenditure", "Baseline family expenditure", "Projected family expenditure"),
        ("poverty_gap", "Baseline income gap", "Projected income gap"),
    ]
    baseline_aggregated = yearly_aggregate(baseline_frame, [spec[0] for spec in metric_specs])
    projection_aggregated = yearly_aggregate(projection_frame, [spec[0] for spec in metric_specs])

    for index, (metric_key, baseline_label, projected_label) in enumerate(metric_specs, start=1):
        row = 1 if index <= 2 else 2
        col = 1 if index % 2 == 1 else 2
        if metric_key in baseline_aggregated.columns:
            fig.add_trace(
                go.Scatter(
                    x=baseline_aggregated["year"],
                    y=baseline_aggregated[metric_key],
                    mode="lines+markers",
                    name=baseline_label,
                ),
                row=row,
                col=col,
            )
        if metric_key in projection_aggregated.columns:
            fig.add_trace(
                go.Scatter(
                    x=projection_aggregated["year"],
                    y=projection_aggregated[metric_key],
                    mode="lines+markers",
                    name=projected_label,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        template="plotly_dark",
        height=720,
        margin=dict(l=30, r=20, t=80, b=30),
        legend_title_text="",
        title="Baseline vs projected yearly trends",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="PHP")
    return fig


def build_dual_metric_figure(frame: pd.DataFrame, title: str, metric_a: str, metric_b: str, label_a: str, label_b: str, y_title: str) -> go.Figure:
    fig = build_dark_figure(title, height=380)
    aggregated = yearly_aggregate(frame, [metric_a, metric_b])
    if not aggregated.empty and metric_a in aggregated.columns:
        fig.add_trace(
            go.Scatter(
                x=aggregated["year"],
                y=aggregated[metric_a],
                mode="lines+markers",
                name=label_a,
            )
        )
    if not aggregated.empty and metric_b in aggregated.columns:
        fig.add_trace(
            go.Scatter(
                x=aggregated["year"],
                y=aggregated[metric_b],
                mode="lines+markers",
                name=label_b,
            )
        )
    if not fig.data:
        add_empty_annotation(fig, "No yearly data available for this chart.")
    fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(title_text="Year")
    return fig


def build_new_poor_figure(frame: pd.DataFrame, title: str) -> go.Figure:
    fig = build_dark_figure(title, height=420)
    aggregated = yearly_aggregate(frame, ["new_poor_flag"])
    if not aggregated.empty and "new_poor_flag" in aggregated.columns:
        fig.add_trace(
            go.Scatter(
                x=aggregated["year"],
                y=aggregated["new_poor_flag"],
                mode="lines+markers",
                name="New poor risk",
            )
        )
        fig.update_yaxes(tickformat=".2%")
    else:
        add_empty_annotation(fig, "No risk data available for this chart.")
    fig.update_yaxes(title_text="Risk rate")
    fig.update_xaxes(title_text="Year")
    return fig


def build_new_poor_comparison_figure(baseline_frame: pd.DataFrame, projection_frame: pd.DataFrame) -> go.Figure:
    fig = build_dark_figure("New poor risk comparison", height=420)
    baseline_aggregated = yearly_aggregate(baseline_frame, ["new_poor_flag"])
    projection_aggregated = yearly_aggregate(projection_frame, ["new_poor_flag"])

    if not baseline_aggregated.empty and "new_poor_flag" in baseline_aggregated.columns:
        fig.add_trace(
            go.Scatter(
                x=baseline_aggregated["year"],
                y=baseline_aggregated["new_poor_flag"],
                mode="lines+markers",
                name="Baseline new poor risk",
            )
        )
    if not projection_aggregated.empty and "new_poor_flag" in projection_aggregated.columns:
        fig.add_trace(
            go.Scatter(
                x=projection_aggregated["year"],
                y=projection_aggregated["new_poor_flag"],
                mode="lines+markers",
                name="Projected new poor risk",
            )
        )

    if fig.data:
        fig.update_yaxes(tickformat=".2%")
    else:
        add_empty_annotation(fig, "No risk data available for this chart.")
    fig.update_yaxes(title_text="Risk rate")
    fig.update_xaxes(title_text="Year")
    return fig


def build_barangay_population_figure(barangay_frame: pd.DataFrame, top_n: int = 10) -> go.Figure:
    fig = build_dark_figure("Pasay barangay population", height=460)
    if barangay_frame.empty or "barangay" not in barangay_frame.columns or "total_population" not in barangay_frame.columns:
        add_empty_annotation(fig, "No barangay totals could be parsed from the census file.")
        return fig

    display_frame = barangay_frame.loc[barangay_frame["barangay"].notna()].copy()
    display_frame = display_frame.sort_values("total_population", ascending=True, na_position="last").tail(top_n)
    if display_frame.empty:
        add_empty_annotation(fig, "No barangay totals could be parsed from the census file.")
        return fig

    fig.add_trace(
        go.Bar(
            x=display_frame["total_population"],
            y=display_frame["barangay"],
            orientation="h",
            marker_color="#7dd3fc",
            name="Total population",
        )
    )
    fig.update_xaxes(title_text="Population")
    fig.update_yaxes(title_text="Barangay")
    return fig


def render_snapshot_section(title: str, frame: pd.DataFrame, metric_specs: list[tuple[str, str, str, bool]]) -> None:
    snapshot = latest_year_snapshot(frame)
    if snapshot.empty:
        st.info(f"No data available for {title.lower()}.")
        return
    build_metric_cards(title, snapshot, metric_specs)


def build_pasay_section(baseline_frame: pd.DataFrame, projection_frame: pd.DataFrame) -> None:
    st.divider()
    st.header("Pasay Modeling and Simulation")
    st.markdown(
        "Pasay City population and barangay-level demographic data provide the local context for the poverty simulation. "
        "The poverty threshold, family income, and newly poor risk are modeled using PSA-based baseline and projection data."
    )
    st.caption("The PC-WAM file is treated as 2020 census-based Pasay demographic data, even though the filename says 2024.")

    city_population_frame = safe_load_csv(
        CITY_POP_PATH,
        rename_map={"city": ["city"], "population": ["pop2026", "population"], "growth_rate": ["growthrate", "growth_rate"]},
        text_columns={"city"},
    )
    city_population_snapshot = extract_latest_city_population(city_population_frame)
    census_context = load_pasay_census_context(PC_WAM_PATH)

    pasay_baseline_frame = extract_pasay_rows(baseline_frame)
    pasay_projection_frame = extract_pasay_rows(projection_frame)
    local_baseline_frame = pasay_baseline_frame if not pasay_baseline_frame.empty else baseline_frame
    local_projection_frame = pasay_projection_frame if not pasay_projection_frame.empty else projection_frame

    if pasay_baseline_frame.empty and pasay_projection_frame.empty:
        st.info("No Pasay-specific poverty rows were found in the CSVs, so the section uses the latest available model snapshots as a fallback.")

    summary_columns = st.columns(4)
    with summary_columns[0]:
        st.metric("Latest Pasay city population", format_population(get_snapshot_value(city_population_snapshot, "population")))
    with summary_columns[1]:
        st.metric("Pasay household population", format_population(census_context["city_total_population"]))
    with summary_columns[2]:
        st.metric("Barangays parsed", format_population(census_context["barangay_count"]))
    with summary_columns[3]:
        projected_risk_source = local_projection_frame if not local_projection_frame.empty else local_baseline_frame
        projected_snapshot = latest_year_snapshot(projected_risk_source)
        st.metric("Latest projected new poor risk", format_percent(get_snapshot_value(projected_snapshot, "new_poor_flag"), ratio=True))

    st.subheader("Pasay demographics breakdown")
    demographic_columns = st.columns(3)
    with demographic_columns[0]:
        st.metric("Total household population", format_population(census_context["city_total_population"]))
    with demographic_columns[1]:
        st.metric("Male population", format_population(census_context["city_male_population"]))
    with demographic_columns[2]:
        st.metric("Female population", format_population(census_context["city_female_population"]))

    demographics_left, demographics_right = st.columns([1, 1])
    with demographics_left:
        st.markdown("**Top age groups**")
        age_groups = census_context["age_groups"]
        if isinstance(age_groups, pd.DataFrame) and not age_groups.empty:
            top_age_groups = age_groups.head(8).copy()
            for column in ["total_population", "male_population", "female_population"]:
                if column in top_age_groups.columns:
                    top_age_groups[column] = top_age_groups[column].apply(format_population)
            st.dataframe(top_age_groups, use_container_width=True, hide_index=True)
        else:
            st.info("No age-group structure could be parsed from the PC-WAM file.")
    with demographics_right:
        st.markdown("**Latest Pasay census context**")
        st.write(
            "The age and sex totals come from the 2020 census-based PC-WAM file. "
            "The city population anchor comes from the 2026 Pasay population file."
        )

    st.subheader("Pasay poverty dynamics")
    poverty_left, poverty_right = st.columns(2)
    with poverty_left:
        st.plotly_chart(
            build_dual_metric_figure(
                local_baseline_frame,
                "Baseline poverty threshold vs family income",
                "poverty_threshold_per_capita",
                "family_income",
                "Baseline poverty threshold",
                "Baseline family income",
                "PHP",
            ),
            use_container_width=True,
        )
    with poverty_right:
        st.plotly_chart(
            build_dual_metric_figure(
                local_projection_frame,
                "Projected poverty threshold vs family income",
                "poverty_threshold_per_capita",
                "family_income",
                "Projected poverty threshold",
                "Projected family income",
                "PHP",
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        build_new_poor_figure(
            local_projection_frame if not local_projection_frame.empty else local_baseline_frame,
            "Projected new poor risk over time",
        ),
        use_container_width=True,
    )

    st.subheader("Barangay summary")
    barangay_frame = census_context["barangay_totals"]
    if isinstance(barangay_frame, pd.DataFrame) and not barangay_frame.empty:
        barangay_table = barangay_frame.loc[:, [column for column in ["barangay", "total_population", "male_population", "female_population"] if column in barangay_frame.columns]].copy()
        for column in ["total_population", "male_population", "female_population"]:
            if column in barangay_table.columns:
                barangay_table[column] = barangay_table[column].apply(format_population)
        st.dataframe(barangay_table, use_container_width=True, hide_index=True)
        st.plotly_chart(build_barangay_population_figure(barangay_frame, top_n=10), use_container_width=True)
    else:
        st.info("No barangay totals were available to display.")


def main():
    st.set_page_config(page_title="Poverty Modeling Dashboard", layout="wide")
    st.title("Poverty Modeling Dashboard")
    st.write(
        "This dashboard compares the latest baseline values from the poverty risk CSV with the latest projected values from the projection CSV. "
        "The new Pasay section adds local population context from the Pasay city and census files."
    )

    view_mode = st.sidebar.radio(
        "View mode",
        ["Compare baseline and projection", "Show only baseline", "Show only projection"],
        index=0,
    )

    baseline_frame = safe_load_csv(RISK_PATH)
    projection_frame = safe_load_csv(PROJECTION_PATH)

    if not baseline_frame.empty and "year" in baseline_frame.columns:
        baseline_frame = baseline_frame.loc[baseline_frame["year"].notna()].reset_index(drop=True)
    if not projection_frame.empty and "year" in projection_frame.columns:
        projection_frame = projection_frame.loc[projection_frame["year"].notna()].reset_index(drop=True)

    projected_metrics = [
        ("Projected poverty threshold", "poverty_threshold_per_capita", "currency", False),
        ("Projected family income", "family_income", "currency", False),
        ("Projected family expenditure", "family_expenditure", "currency", False),
        ("Projected income gap", "poverty_gap", "currency", False),
        ("Projected new poor risk", "new_poor_flag", "percent", True),
    ]
    baseline_metrics = [
        ("Baseline poverty threshold", "poverty_threshold_per_capita", "currency", False),
        ("Baseline family income", "family_income", "currency", False),
        ("Baseline family expenditure", "family_expenditure", "currency", False),
        ("Baseline income gap", "poverty_gap", "currency", False),
        ("Baseline new poor risk", "new_poor_flag", "percent", True),
    ]

    if baseline_frame.empty and projection_frame.empty:
        st.warning("The poverty CSV files could not be loaded, so only the Pasay context section below will be available.")
    else:
        if view_mode in {"Compare baseline and projection", "Show only projection"}:
            render_snapshot_section("Latest projected values", projection_frame, projected_metrics)

        if view_mode in {"Compare baseline and projection", "Show only baseline"}:
            st.divider()
            render_snapshot_section("Latest baseline values", baseline_frame, baseline_metrics)

        if view_mode == "Compare baseline and projection":
            st.divider()
            st.subheader("Baseline vs projection comparison")
            comparison_figure = build_comparison_figure(baseline_frame, projection_frame)
            st.plotly_chart(comparison_figure, use_container_width=True)

            st.subheader("New poor risk comparison")
            new_poor_figure = build_new_poor_comparison_figure(baseline_frame, projection_frame)
            st.plotly_chart(new_poor_figure, use_container_width=True)

        elif view_mode == "Show only baseline":
            st.divider()
            st.subheader("Baseline trends")
            baseline_only_figure = build_new_poor_figure(baseline_frame, "Baseline new poor risk")
            st.plotly_chart(baseline_only_figure, use_container_width=True)

        elif view_mode == "Show only projection":
            st.divider()
            st.subheader("Projection trends")
            projection_only_figure = build_new_poor_figure(projection_frame, "Projected new poor risk")
            st.plotly_chart(projection_only_figure, use_container_width=True)

    build_pasay_section(baseline_frame, projection_frame)


if __name__ == "__main__":
    main()