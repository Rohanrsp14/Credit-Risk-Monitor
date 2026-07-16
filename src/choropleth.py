"""Interactive Texas county choropleth: approval rate, volume, minority share.

Input:  data/processed/hmda_features.csv
        data/raw/tx_counties.geojson (fetched once, cached locally)
Output: outputs/texas_risk_heatmap.html

County boundary source: US Census-derived, FIPS-keyed public GeoJSON from
the plotly/datasets GitHub repo
(https://github.com/plotly/datasets/blob/master/geojson-counties-fips.json).
"""

from pathlib import Path

import folium
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"

HMDA_FEATURES_PATH = PROCESSED_DIR / "hmda_features.csv"
TX_GEOJSON_PATH = RAW_DIR / "tx_counties.geojson"
HEATMAP_PATH = OUTPUTS_DIR / "texas_risk_heatmap.html"

COUNTIES_GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
TX_STATE_FIPS = "48"

# Texas roughly centered for the initial map view.
TX_MAP_CENTER = [31.0, -99.3]
TX_MAP_ZOOM = 6


def get_tx_geojson():
    if TX_GEOJSON_PATH.exists():
        print(f"Using cached county boundaries: {TX_GEOJSON_PATH}")
        return _read_geojson(TX_GEOJSON_PATH)

    print(f"Fetching county boundaries from {COUNTIES_GEOJSON_URL} ...")
    resp = requests.get(COUNTIES_GEOJSON_URL, timeout=60)
    resp.raise_for_status()
    all_counties = resp.json()

    tx_features = [f for f in all_counties["features"] if f["properties"]["STATE"] == TX_STATE_FIPS]
    tx_geojson = {"type": "FeatureCollection", "features": tx_features}
    print(f"Filtered to {len(tx_features)} Texas counties")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    import json

    with open(TX_GEOJSON_PATH, "w") as f:
        json.dump(tx_geojson, f)
    print(f"Cached to {TX_GEOJSON_PATH}")

    return tx_geojson


def _read_geojson(path):
    import json

    with open(path) as f:
        return json.load(f)


def build_county_stats():
    # county_fips must be read as str: it's an all-numeric-looking column
    # ("48201"), so pandas' default dtype inference silently reads it as
    # float64 (48201.0), which later stringifies to "48201.0" and fails to
    # join against the GeoJSON's zero-padded 5-char string ids ("48201").
    df = pd.read_csv(HMDA_FEATURES_PATH, low_memory=False, dtype={"county_fips": str})
    df = df.dropna(subset=["county_fips"]).copy()
    df["county_fips"] = df["county_fips"].str.zfill(5)

    stats = (
        df.groupby("county_fips")
        .agg(
            application_count=("approved", "count"),
            approval_rate=("approved", "mean"),
            minority_share=("is_minority", "mean"),
        )
        .reset_index()
    )
    print(f"Built county-level stats for {len(stats)} counties")
    return stats


def build_choropleth(stats, geojson, county_names):
    stats = stats.merge(county_names, on="county_fips", how="left")

    m = folium.Map(location=TX_MAP_CENTER, zoom_start=TX_MAP_ZOOM, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=geojson,
        data=stats,
        columns=["county_fips", "approval_rate"],
        key_on="feature.id",
        fill_color="RdYlGn",
        fill_opacity=0.8,
        line_opacity=0.3,
        legend_name="HMDA Approval Rate (2022, TX)",
        nan_fill_color="lightgrey",
    ).add_to(m)

    stats_lookup = stats.set_index("county_fips").to_dict("index")
    for feature in geojson["features"]:
        fips = feature["id"]
        row = stats_lookup.get(fips)
        if row is None:
            feature["properties"]["tooltip"] = f"{feature['properties']['NAME']} County: no data"
        else:
            feature["properties"]["tooltip"] = (
                f"{row.get('county_name', feature['properties']['NAME'])} County | "
                f"Approval rate: {row['approval_rate']:.1%} | "
                f"Applications: {row['application_count']:,} | "
                f"Minority share: {row['minority_share']:.1%}"
            )

    folium.GeoJson(
        geojson,
        style_function=lambda x: {"fillOpacity": 0, "color": "transparent"},
        tooltip=folium.GeoJsonTooltip(fields=["tooltip"], aliases=[""], labels=False),
    ).add_to(m)

    return m


def run():
    geojson = get_tx_geojson()
    stats = build_county_stats()

    county_names = pd.DataFrame(
        [{"county_fips": f["id"], "county_name": f["properties"]["NAME"]} for f in geojson["features"]]
    )

    m = build_choropleth(stats, geojson, county_names)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    m.save(str(HEATMAP_PATH))
    print(f"Saved choropleth map to {HEATMAP_PATH}")

    return m, stats


if __name__ == "__main__":
    run()
