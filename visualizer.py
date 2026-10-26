from pathlib import Path
from urllib.request import urlopen
import inspect
import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Configure the page before displaying other Streamlit elements.
st.set_page_config(
    page_title="The Backpack Doctor",
    page_icon="🎒",
    layout="wide",
)

# Apply consistent spacing and styling to the dashboard.
st.markdown(
    """
    <style>
    /* Keep the content readable on wide screens. */
    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Give headings a consistent appearance. */
    h1 {
        letter-spacing: -0.035em;
        color: #183044;
    }

    h2, h3 {
        color: #183044;
        letter-spacing: -0.02em;
    }

    /* Style the existing bordered result containers. */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 16px;
    }

    /* Give metrics more breathing room. */
    [data-testid="stMetric"] {
        padding: 0.75rem 0;
    }

    /* Make the sidebar submit button fill its available width. */
    [data-testid="stSidebar"]
    [data-testid="stFormSubmitButton"] button {
        width: 100%;
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
    }

    /* Hide numeric increment/decrement buttons. */
    [data-testid="stSidebar"]
    [data-testid="stNumberInput"] button {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Project files exported by the modelling notebook.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "triage_model.joblib"
COUNTY_PATH = PROJECT_DIR / "data" / "processed" / "rq2_county_ppv.csv"
SUBGROUP_PATH = (
    PROJECT_DIR / "data"/ "processed" / "rq3_subgroup_performance.csv"
)

LOCAL_BOUNDARIES = PROJECT_DIR / "data" / "geo" / "counties.geojson"

BOUNDARY_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/"
    "master/geojson-counties-fips.json"
)


STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona",
    "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
    "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico",
}


# File versions invalidate caches when project exports change.
def file_version(path):
    info = path.stat()
    return info.st_mtime_ns, info.st_size


# Load the fitted model once instead of on every interaction.
@st.cache_resource(show_spinner=False)
def load_model_bundle(path, version):
    bundle = joblib.load(path)
    required = {"model", "features", "threshold"}
    missing = required - set(bundle)

    if missing:
        raise ValueError(
            "Missing model information: "+ ", ".join(sorted(missing)))

    if not 0 <= float(bundle["threshold"]) <= 1:
        raise ValueError( "The model threshold must be between 0 and 1.")
    return bundle


# Load and prepare the county results once per file version.
@st.cache_data(show_spinner=False)
def load_county_results(path, version):
    counties = pd.read_csv(path,dtype={"fips": "string"},)

    required = {
        "fips",
        "locationname",
        "stateabbr",
        "diagnosed_diabetes_pct",
        "screening_prevalence",
        "PPV",
        "NPV",
    }

    missing = required - set(counties.columns)

    if missing:
        raise ValueError(
            "Missing county columns: " + ", ".join(sorted(missing)))

    # Preserve leading zeros in county identifiers.
    counties["fips"] = (
        counties["fips"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    counties["stateabbr"] = (
        counties["stateabbr"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    for column in [
        "diagnosed_diabetes_pct",
        "screening_prevalence",
        "PPV",
        "NPV",
    ]:
        counties[column] = pd.to_numeric(
            counties[column],
            errors="coerce",
        )

    original_count = len(counties)

    counties = counties.dropna(
        subset=["fips", "locationname", "stateabbr", "PPV"]
    ).copy()

    excluded_count = original_count - len(counties)

    if counties.empty:
        raise ValueError("No usable county results were found.")
    if not counties["fips"].str.fullmatch(r"\d{5}").all():
        raise ValueError("County FIPS identifiers must contain five digits." )
    if counties["fips"].duplicated().any():
        raise ValueError("The county table contains duplicate FIPS identifiers.")
    if not counties["PPV"].between(0, 1).all():
        raise ValueError("PPV must be stored as a fraction between 0 and 1.")

    counties["PPV_percent"] = counties["PPV"] * 100

    counties["county_label"] = (counties["locationname"] + ", " + counties["stateabbr"])

    return counties, excluded_count


# Preserve subgroup results exactly as exported.
@st.cache_data(show_spinner=False)
def load_subgroup_results(path, version):
    subgroups = pd.read_csv(path)

    required = {
        "dimension",
        "group",
        "n",
        "positives",
        "sensitivity",
        "sens low",
        "sens high",
        "specificity",
        "PPV",
        "missed cases (FN)",
        "interpretable",
    }

    missing = required - set(subgroups.columns)
    if missing:
        raise ValueError(
            "Missing subgroup columns: "
            + ", ".join(sorted(missing))
        )

    for column in ("dimension", "group"):
        subgroups[column] = (
            subgroups[column].astype("string").str.strip()
        )

    numeric_columns = [
        "n",
        "positives",
        "sensitivity",
        "sens low",
        "sens high",
        "specificity",
        "PPV",
        "missed cases (FN)",
    ]

    for column in numeric_columns:
        subgroups[column] = pd.to_numeric(
            subgroups[column],
            errors="coerce",
        )

    def parse_interpretable(value):
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1"}
        return pd.notna(value) and value == True

    subgroups["interpretable"] = subgroups["interpretable"].map(
        parse_interpretable
    )

    if subgroups.duplicated(["dimension", "group"]).any():
        raise ValueError(
            "The subgroup table has duplicate dimension/group records."
        )

    return subgroups

def format_percentage(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Not available"
    return f"{number:.1%}" if 0 <= number <=1 else "Not available"

def format_count(value):
    try:
        number = float(value)
    except(TypeError, ValueError):
        return "Not available"
    return f"{number:,.0f}" if pd.notna(number) else "Not available"

@st.cache_resource(show_spinner=False)
def load_boundary_index(source, version):
    if source.startswith("https://"):
        with urlopen(source, timeout=20) as response:
            boundaries = json.load(response)
    else:
        with open(source, encoding="utf-8") as handle:
            boundaries = json.load(handle)

    features = boundaries.get("features")
    if boundaries.get("type") != "FeatureCollection" or not features:
        raise ValueError("The map requires a GeoJSON FeatureCollection.")

    index = {}

    for feature in features:
        properties = feature.get("properties") or {}
        raw_id = feature.get("id")

        if raw_id is None:
            for key in ("GEOID", "GEOID10", "GEO_ID", "FIPS", "fips"):
                if properties.get(key) is not None:
                    raw_id = properties[key]
                    break

        if (
            raw_id is None
            and properties.get("STATEFP") is not None
            and properties.get("COUNTYFP") is not None
        ):
            raw_id = (
                f"{properties['STATEFP']}{properties['COUNTYFP']}"
            )

        if raw_id is None:
            continue

        raw_id = str(raw_id).strip()
        if raw_id.endswith(".0"):
            raw_id = raw_id[:-2]

        digits = "".join(char for char in raw_id if char.isdigit())
        if not digits:
            continue

        fips = digits[-5:].zfill(5)

        normalized_feature = dict(feature)
        normalized_feature["id"] = fips
        index[fips] = normalized_feature

    if not index:
        raise ValueError("No county FIPS identifiers were found in the GeoJSON.")

    return index

# Prefer archived boundaries when they are available.
def get_boundary_source():
    if LOCAL_BOUNDARIES.exists():
        return str(LOCAL_BOUNDARIES),file_version(LOCAL_BOUNDARIES)
    return BOUNDARY_URL, "remote"


# The shared boundary index is only read, never modified.
@st.cache_resource(show_spinner=False)
def build_base_map(map_data, source_version, national_view, _boundary_index):
    geometry = {
        "type": "FeatureCollection",
        "features": [
            _boundary_index[fips]
            for fips in map_data["fips"]
            if fips in _boundary_index
        ],
    }
    figure = px.choropleth(
        map_data,
        geojson=geometry,
        featureidkey="id",
        locations="fips",
        color="PPV_percent",
        scope="usa",
        color_continuous_scale="Blues",
        range_color=(0, 100),
        hover_name="county_label",
        hover_data={
            "fips": True,
            "PPV_percent": ":.1f",
            "diagnosed_diabetes_pct": ":.1f",
        },
        labels={
            "fips": "County FIPS",
            "PPV_percent": "Estimated positive predictive value (%)",
            "diagnosed_diabetes_pct": "Diagnosed diabetes prevalence (%)",
        },
    )
    figure.update_traces(
        marker_line_width=0.25,
        marker_line_color="#FFFFFF",
    )
    if not national_view:
        figure.update_geos(fitbounds="locations")
    figure.update_geos(bgcolor="rgba(0,0,0,0)", showlakes=False)
    figure.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title="PPV (%)",
            thickness=12,
            len=0.7,
        ),
        uirevision=(
            "US"
            if national_view
            else str(map_data["stateabbr"].iloc[0])
        ),
    )
    return figure



# Support both older and newer Streamlit chart-width arguments.
def display_map(figure):
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(
            figure,
            width="stretch",
            key="county_map",
        )
    else:
        st.plotly_chart(
            figure,
            use_container_width=True,
            key="county_map",
        )


# PANEL 1 - NEW VERSION

# Collect all inputs in one form and publish results only after submission.
def show_patient_prediction(bundle, model_version, counties, subgroup_data):
    if st.session_state.get("model_version") != model_version:
        st.session_state.pop("patient_prediction", None)
        st.session_state.pop("dashboard_selection", None)
        st.session_state["model_version"] = model_version

    with st.sidebar:
        st.header("Inputs")
        st.caption(
            "Review the patient and dashboard selections, then submit once."
        )

        with st.form("all_inputs_form"):
            st.subheader("Patient measurements")

            age = st.number_input(
                "Age (years)",
                min_value=20,
                max_value=100,
                value=30,
                step=1,
            )
            sex = st.selectbox("Sex", ("Male", "Female"))
            waist = st.number_input(
                "Waist circumference (cm)",
                min_value=40.0,
                max_value=200.0,
                value=90.0,
                step=0.5,
            )
            systolic = st.number_input(
                "Systolic blood pressure (mmHg)",
                min_value=60.0,
                max_value=260.0,
                value=120.0,
                step=1.0,
            )
            diastolic = st.number_input(
                "Diastolic blood pressure (mmHg)",
                min_value=30.0,
                max_value=160.0,
                value=80.0,
                step=1.0,
            )

            st.subheader("Dashboard inputs")

            if counties is None or counties.empty:
                place_label = st.selectbox(
                    "State / County",
                    ["County data unavailable"],
                    disabled=True,
                )
                place_options = {}
            else:
                place_options = {
                    (
                        f"{STATE_NAMES.get(row.stateabbr, row.stateabbr)} · "
                        f"{row.locationname} ({row.fips})"
                    ): (row.stateabbr, row.fips)
                    for row in counties[
                        ["stateabbr", "locationname", "fips"]
                    ].drop_duplicates().itertuples(index=False)
                }
                place_label = st.selectbox(
                    "State / County",
                    sorted(place_options),
                )

            race_options = []
            if (
                subgroup_data is not None
                and {"dimension", "group"}.issubset(subgroup_data.columns)
            ):
                race_options = sorted(
                    set(
                        subgroup_data.loc[
                            subgroup_data["dimension"] == "race",
                            "group",
                        ].dropna().astype(str)
                    ) - {"Not specified"}
                )

            selected_race = st.selectbox(
                "Ethnicity / race group",
                ["Not specified"] + race_options,
                disabled=subgroup_data is None,
            )

            submitted = st.form_submit_button(
                "Confirm inputs and update KPIs",
                type="primary",
            )

    if submitted:
        st.session_state.pop("patient_prediction", None)
        st.session_state.pop("dashboard_selection", None)

        patient = pd.DataFrame([{
            "age": age,
            "female": int(sex == "Female"),
            "systolic": systolic,
            "diastolic": diastolic,
            "waist": waist,
        }])

        missing_features = set(bundle["features"]) - set(patient.columns)

        if missing_features:
            st.sidebar.error(
                "Required model inputs are missing: "
                + ", ".join(sorted(missing_features))
            )
        elif systolic <= diastolic:
            st.sidebar.error(
                "Please check blood pressure: systolic pressure must exceed "
                "diastolic pressure."
            )
        else:
            try:
                model = bundle["model"]
                positive_class_index = list(model.classes_).index(1)
                risk = float(
                    model.predict_proba(
                        patient[bundle["features"]]
                    )[0, positive_class_index]
                )

                if not 0 <= risk <= 1:
                    raise ValueError(
                        "The model returned an invalid probability."
                    )

            except Exception as error:
                st.sidebar.error(
                    f"The prediction could not be calculated: {error}"
                )
            else:
                st.session_state["patient_prediction"] = {
                    "risk": risk,
                    "threshold": float(bundle["threshold"]),
                    "measurements": patient.iloc[0].to_dict(),
                    "sex": sex,
                }

                selected_state, selected_fips = place_options.get(
                    place_label,
                    (None, None),
                )
                st.session_state["dashboard_selection"] = {
                    "selected_state": selected_state,
                    "selected_fips": selected_fips,
                    "selected_race": selected_race,
                }

    return st.session_state.get(
        "dashboard_selection",
        {
            "selected_state": None,
            "selected_fips": None,
            "selected_race": "Not specified",
        },
    )

def patient_subgroup_rows(subgroups, prediction, selected_race):
    age = int(prediction["measurements"]["age"])

    if age < 40:
        age_band = "20-39"
    elif age < 55:
        age_band = "40-54"
    elif age < 70:
        age_band = "55-69"
    else:
        age_band = "70+"

    comparisons = [
        ("Sex", "sex", prediction["sex"]),
        ("Age band", "age_band", age_band),
    ]

    if selected_race != "Not specified":
        comparisons.append(
            ("Race/ethnicity", "race", selected_race)
        )

    rows = []

    for label, dimension, group in comparisons:
        match = subgroups.loc[
            (subgroups["dimension"] == dimension)
            & (subgroups["group"] == group)
        ]

        if match.empty:
            rows.append({
                "dimension": label,
                "group": group,
                "sensitivity": "Unavailable",
                "ci": "—",
                "positives": "Unavailable",
            })
            continue

        row = match.iloc[0]

        if not bool(row["interpretable"]):
            sensitivity = "Not interpretable"
            ci = "—"
        else:
            sensitivity = format_percentage(row["sensitivity"])
            low = format_percentage(row["sens low"])
            high = format_percentage(row["sens high"])
            ci = (
                f"{low}–{high}"
                if low != "Not available" and high != "Not available"
                else "Not available"
            )

        rows.append({
            "dimension": label,
            "group": group,
            "sensitivity": sensitivity,
            "ci": ci,
            "positives": format_count(row["positives"]),
        })

    return rows

def show_kpi_overview(
    counties,
    selected_state,
    selected_fips,
    subgroup_data,
    selected_race,
):
    st.divider()
    st.subheader("Triage overview")

    rq1, rq2, rq3 = st.columns(3)

    with rq1:
        with st.container(border=True):
            st.markdown("**RQ1 · Individual screening**")
            result = st.session_state.get("patient_prediction")

            if result is None:
                st.metric("Estimated A1C ≥ 5.7% probability", "—")
                st.caption(
                    "Enter patient measurements in the sidebar "
                    "and calculate a prediction."
                )
            else:
                risk = result["risk"]
                threshold = result["threshold"]
                st.metric(
                    "Estimated A1C ≥ 5.7% probability",
                    f"{risk:.1%}",
                )
                if risk >= threshold:
                    st.warning(
                        "Screening positive: at or above the model "
                        "threshold. Consider laboratory A1C follow-up."
                    )
                else:
                    st.info(
                        "Below the model screening threshold. "
                        "This does not rule out prediabetes or diabetes."
                    )

                st.caption(f"Model threshold: {threshold:.1%}.")

    with rq2:
        with st.container(border=True):
            st.markdown("**RQ2 · County reliability**")

            if counties is None or selected_fips is None:
                st.metric("Estimated county PPV", "—")
                st.caption("County results are unavailable.")
            else:
                match = counties[
                    (counties["stateabbr"] == selected_state)
                    & (counties["fips"] == selected_fips)
                ]

                if match.empty:
                    st.metric("Estimated county PPV", "—")
                    st.caption("No result is available for this county.")
                else:
                    county = match.iloc[0]
                    ppv = float(county["PPV"])
                    st.metric("Estimated county PPV", f"{ppv:.1%}")
                    st.write(
                        f"**Response:** Under the RQ2 assumptions, about "
                        f"{ppv * 100:.0f} of 100 positive flags in "
                        f"{county['locationname']} would correspond to "
                        "A1C ≥ 5.7%."
                    )
                    st.caption(
                        "This is an assumption-based estimate, "
                        "not a local validation."
                    )

    with rq3:
        with st.container(border=True):
            st.markdown("**RQ3 · Similar patient groups**")

            required = {
                "dimension", "group", "positives",
                "sensitivity", "sens low", "sens high",
                "interpretable",
            }

            prediction = st.session_state.get("patient_prediction")

            if subgroup_data is None or not required.issubset(
                subgroup_data.columns
            ):
                st.caption("Subgroup results are unavailable.")
            elif prediction is None:
                st.caption(
                    "Enter and confirm patient measurements to show "
                    "matching subgroup results."
                )
            else:
                rows = patient_subgroup_rows(
                    subgroup_data,
                    prediction,
                    selected_race,
                )

                for row in rows:
                    st.write(
                        f"**{row['dimension']}: {row['group']}** — "
                        f"Sensitivity {row['sensitivity']}; "
                        f"95% CI {row['ci']}; "
                        f"{row['positives']} positive cases."
                    )

                if selected_race == "Not specified":
                    st.caption(
                        "Select a race/ethnicity group to include it."
                    )

# Panel 2: map and numerical results from the county reliability export.

def show_county_detail_card(selected_county, selected_state):
    state_name = STATE_NAMES.get(selected_state, selected_state)

    st.markdown(
        f"**{selected_county['locationname']}, {state_name}**"
    )

    ppv = float(selected_county["PPV"])

    st.metric(
        "Estimated PPV of a positive flag",
        f"{ppv:.1%}",
    )

    st.info(
        f"Under the RQ2 assumptions, about "
        f"{ppv * 100:.0f} out of 100 people flagged "
        "positive would have A1C ≥ 5.7%."
    )

    st.caption(
        "County PPV describes positive screening flags "
        "in the region. It does not replace the "
        "patient's individual risk estimate."
    )

    with st.expander("Prevalence assumptions"):
        diagnosed_pct = selected_county["diagnosed_diabetes_pct"]
        screening_prevalence = selected_county["screening_prevalence"]

        st.metric(
            "Diagnosed diabetes · PLACES",
            (
                f"{diagnosed_pct:.1f}%"
                if pd.notna(diagnosed_pct)
                else "Unavailable"
            ),
        )

        st.metric(
            "Assumed prevalence · A1C ≥ 5.7%",
            (
                f"{screening_prevalence:.1%}"
                if pd.notna(screening_prevalence)
                and 0 <= screening_prevalence <= 1
                else "Unavailable"
            ),
        )

        st.caption(
            "The target prevalence is derived in RQ2. "
            "It is not directly measured by PLACES."
        )


def draw_county_map(
    matched,
    selected_fips,
    boundary_index,
    source,
    version,
    national_view,
):
    map_data = matched[
        [
            "fips",
            "stateabbr",
            "PPV_percent",
            "county_label",
            "diagnosed_diabetes_pct",
        ]
    ]

    figure = build_base_map(
        map_data,
        (source, version),
        national_view,
        boundary_index,
    )

    # Copy the cached map before adding the selection outline.
    figure = go.Figure(figure.to_dict())

    selected_feature = boundary_index.get(selected_fips)

    if selected_feature is not None:
        figure.add_trace(
            go.Choropleth(
                geojson={
                    "type": "FeatureCollection",
                    "features": [selected_feature],
                },
                featureidkey="id",
                locations=[selected_fips],
                z=[0],
                zmin=0,
                zmax=1,
                colorscale=[
                    [0, "rgba(0,0,0,0)"],
                    [1, "rgba(0,0,0,0)"],
                ],
                marker_line_color="#E87924",
                marker_line_width=2.5,
                showscale=False,
                hoverinfo="skip",
            )
        )

    display_map(figure)


def show_county_map(counties, state_counties, selected_fips):
    national_view = st.checkbox(
        "US overview (slower)",
        value=False,
        key="national_map",
    )

    show_map = st.checkbox(
        "Show interactive map",
        value=True,
        key="show_map",
    )

    if not show_map:
        return

    try:
        source, version = get_boundary_source()

        with st.spinner("Preparing county map…"):
            boundary_index = load_boundary_index(source, version)

            view = counties if national_view else state_counties

            matched = view[
                view["fips"].isin(boundary_index)
            ].copy()

            unmatched = view[
                ~view["fips"].isin(boundary_index)
            ]

            if matched.empty:
                st.info(
                    "No matching boundaries are "
                    "available for this map view."
                )
            else:
                draw_county_map(
                    matched,
                    selected_fips,
                    boundary_index,
                    source,
                    version,
                    national_view,
                )

        if not unmatched.empty:
            st.caption(
                f"{len(unmatched)} counties in this "
                "view have no matching boundary. "
                "Their results remain available "
                "through the county selector."
            )

            with st.expander("Counties without matching boundaries"):
                st.dataframe(
                    unmatched[
                        ["fips", "locationname", "stateabbr"]
                    ],
                    hide_index=True,
                )

        st.caption(
            "Hover to inspect a county. The orange outline "
            "marks the dropdown selection. Map clicks "
            "do not change the selected county."
        )

    except Exception as error:
        st.info(
            "The map is temporarily unavailable. "
            "County results remain available on the right."
        )

        with st.expander("Map loading details"):
            st.code(str(error))


def show_county_assumptions(counties, selected_county):
    st.write(
        "County PPV combines evaluated sensitivity and specificity "
        "with an assumed local target prevalence. It assumes model "
        "performance transfers to the county. This is not a local "
        "validation of the model."
    )

    st.write(
        "PLACES measures diagnosed diabetes. The RQ2 analysis "
        "converts this to an assumed prevalence of A1C ≥ 5.7%."
    )

    additional_columns = [
        column
        for column in counties.columns
        if column.startswith(("PPV_", "NPV_"))
        and column != "PPV_percent"
    ]

    if additional_columns:
        st.caption(
            "Additional PPV and NPV scenario estimates; "
            "values are shown as percentages."
        )

        additional_values = selected_county[additional_columns].map(
            format_percentage
        )

        st.table(
            additional_values.rename("Estimated value").to_frame()
        )

def show_county_reliability(
    counties,
    excluded_count,
    selected_state,
    selected_fips,
):
    if selected_state is None or selected_fips is None:
        st.info(
            "Confirm a state and county in the sidebar "
            "to view regional results."
        )
        return

    if counties is None or counties.empty:
        st.info("County results are unavailable.")
        return

    st.divider()
    st.subheader("2 · Regional reliability")
    st.write(
        "Explore how often a positive screening flag is expected "
        "to be correct in the selected county."
    )

    if excluded_count:
        st.caption(
            f"{excluded_count} rows with missing identifiers or PPV "
            "were excluded. Missing values are not treated as zero."
        )

    state_counties = counties.loc[
        counties["stateabbr"] == selected_state
    ].copy()

    selected_rows = state_counties.loc[
        state_counties["fips"] == selected_fips
    ]

    if selected_rows.empty:
        st.warning("The selected county has no reliability result.")
        return

    selected_county = selected_rows.iloc[0]

    with st.container(border=True):
        map_column, detail_column = st.columns([2.4, 1])

        with detail_column:
            show_county_detail_card(
                selected_county,
                selected_state,
            )

        with map_column:
            show_county_map(
                counties,
                state_counties,
                selected_fips,
            )

    with st.expander("RQ2 assumptions and additional results"):
        show_county_assumptions(counties, selected_county)


# Panel 3: match the submitted patient to separately evaluated groups.
def show_subgroup_performance(subgroups, error=None):
    st.divider()
    st.subheader("3"
                 " · Performance for comparable patient groups")

    if error:
        st.error(error)
        return

    if subgroups is None:
        st.warning("Subgroup results are unavailable.")
        return

    # Format unavailable or invalid metrics without inventing values.
    def percentage(value):
        if pd.notna(value) and 0 <= value <= 1:
            return f"{value:.1%}"
        return "Not available"

    def count(value):
        if pd.notna(value):
            return f"{value:,.0f}"
        return "Not available"

    display = subgroups.copy()
    display["Dimension"] = display["dimension"].replace({
        "sex": "Sex",
        "race": "Race / ethnicity",
        "age_band": "Age band",
    })
    display["N"] = display["n"].map(count)
    display["Positive cases"] = display["positives"].map(count)

    def sensitivity_ci(row):
        if not bool(row["interpretable"]):
            return "Not interpreted"

        values = [
            percentage(row["sensitivity"]),
            percentage(row["sens low"]),
            percentage(row["sens high"]),
        ]

        if any(value == "Not available" for value in values):
            return "Not available"

        return f"{values[0]} ({values[1]}–{values[2]})"

    display["Sensitivity (95% CI)"] = display.apply(
        sensitivity_ci,
        axis=1,
    )
    display["Specificity"] = display["specificity"].map(percentage)
    display["Missed positive cases"] = display[
        "missed cases (FN)"
    ].map(count)

    display = display.rename(columns={
        "group": "Group",
        "interpretable": "Interpretable",
    })

    columns = [
        "Dimension",
        "Group",
        "N",
        "Positive cases",
        "Sensitivity (95% CI)",
        "Specificity",
        "Missed positive cases",
        "Interpretable",
    ]

    auc_columns = {"AUC", "AUC low", "AUC high"}
    if auc_columns.issubset(display.columns):
        for column in auc_columns:
            display[column] = pd.to_numeric(
                display[column],
                errors="coerce",
            )

        def auc_ci(row):
            values = [row["AUC"], row["AUC low"], row["AUC high"]]
            if any(pd.isna(value) for value in values):
                return "Not available"
            return (
                f"{values[0]:.3f} "
                f"({values[1]:.3f}–{values[2]:.3f})"
            )

        display["AUC (95% CI)"] = display.apply(auc_ci, axis=1)
        columns.insert(
            columns.index("Missed positive cases"),
            "AUC (95% CI)",
        )

    if st.session_state.get("patient_prediction") is not None:
        st.caption(
            "The KPI above shows results for the submitted patient's "
            "matching groups. This table compares all groups."
        )
    else:
        st.caption("This table compares all evaluated subgroup results.")

    st.table(display.loc[:, columns])


def prepare_subgroup_results():
    try:
        subgroups = load_subgroup_results(
            str(SUBGROUP_PATH),
            file_version(SUBGROUP_PATH),
        )
        return subgroups, None
    except Exception as error:
        return None, f"Subgroup results could not be loaded: {error}"


def main():
    st.title("The Backpack Doctor")
    st.caption(
        "FIELD-BASED DYSGLYCEMIA TRIAGE · "
        "NHANES 2021–2023 · CDC PLACES"
    )
    st.write(
        "Patient risk, regional reliability and performance across "
        "patient groups. Start with the inputs and KPI answers below; "
        "the detailed map and subgroup results follow."
    )

    bundle = None
    model_version = None

    try:
        model_version = file_version(MODEL_PATH)
        bundle = load_model_bundle(
            str(MODEL_PATH),
            model_version,
        )
    except Exception as error:
        st.session_state.pop("patient_prediction", None)
        st.session_state.pop("model_version", None)
        st.error(f"The fitted model could not be loaded: {error}")

    counties = None
    excluded_count = 0
    county_error = None

    try:
        counties, excluded_count = load_county_results(
            str(COUNTY_PATH),
            file_version(COUNTY_PATH),
        )
    except Exception as error:
        county_error = str(error)


    with st.expander("Model and evaluation"):
        st.write(
            "The deployed model was refitted on the full analytic sample. "
            "Reported model performance comes from the evaluation in the "
            "modelling notebook, not from evaluating the refitted model on "
            "its training data."
        )
        st.write(
            "The dashboard loads the fitted model and exported county and "
            "subgroup results. It does not retrain the model or repeat the "
            "evaluation."
        )

    subgroups, subgroup_error = prepare_subgroup_results()

    if bundle is not None:
        dashboard_selection = show_patient_prediction(
            bundle,
            model_version,
            counties,
            subgroups,
        )
    else:
        dashboard_selection = {
            "selected_state": None,
            "selected_fips": None,
            "selected_race": "Not specified",
        }

    selected_state = dashboard_selection["selected_state"]
    selected_fips = dashboard_selection["selected_fips"]
    selected_race = dashboard_selection["selected_race"]

    tab_overview, tab_regional, tab_subgroups = st.tabs(
        [
            "1 · Triage overview",
            "2 · Regional reliability",
            "3 · Subgroup performance",
        ]
    )

    with tab_overview:
        show_kpi_overview(
            counties,
            selected_state,
            selected_fips,
            subgroups,
            selected_race,
        )

    with tab_regional:
        if counties is None:
            st.warning(
                f"County results could not be loaded: {county_error}"
            )
        else:
            show_county_reliability(
                counties,
                excluded_count,
                selected_state,
                selected_fips,
            )

    with tab_subgroups:
        show_subgroup_performance(
            subgroups,
            subgroup_error,
        )

    st.divider()
    st.caption(
        "HSLU · MSc Applied Data Science & AI · "
        "CIP02 · Research dashboard"
    )


if __name__ == "__main__":
    main()