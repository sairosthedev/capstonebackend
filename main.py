"""
Farm Advisory API — FastAPI backend for ML + rule engine
Suppresses sklearn warnings, uses numpy arrays for compatibility.
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Zimbabwe places — towns, districts, provinces (expandable)
ZIMBABWE_PLACES = {
    # Major cities
    "harare": (-17.828, 31.053), "bulawayo": (-20.15, 28.583), "mutare": (-18.971, 32.671),
    "gweru": (-19.45, 29.817), "chitungwiza": (-18.013, 31.076), "kwekwe": (-18.928, 29.815),
    "kadoma": (-18.333, 29.915), "masvingo": (-20.064, 30.828), "chinhoyi": (-17.367, 30.2),
    "epworth": (-17.889, 31.147), "marondera": (-18.185, 31.552), "nyanga": (-18.217, 32.750),
    "chiredzi": (-21.05, 31.667), "zvishavane": (-20.333, 30.067), "bindura": (-17.301, 31.330),
    "beithridge": (-22.217, 30.000), "beitbridge": (-22.217, 30.000),
    "kariba": (-16.533, 28.800), "victoria falls": (-17.933, 25.867), "hwange": (-18.367, 26.500),
    # Provinces & districts
    "gwanda": (-20.933, 29.0), "lupane": (-18.933, 27.8), "chipinge": (-20.2, 32.617),
    "rusape": (-18.533, 32.133), "mutoko": (-17.4, 32.217), "makoni": (-18.117, 32.25),
    "chegutu": (-18.133, 30.15), "norton": (-17.883, 30.7), "zvimba": (-17.867, 30.4),
    "gokwe": (-18.217, 28.933), "mvuma": (-19.283, 30.533), "gutu": (-19.65, 31.017),
    "chivi": (-20.083, 30.883), "mwenezi": (-21.033, 30.75), "kezi": (-21.15, 29.467),
    "umguza": (-20.15, 28.55), "binga": (-16.933, 29.267), "nyamandlovu": (-19.967, 28.533),
    "hwedza": (-18.65, 31.667), "murehwa": (-17.65, 31.783), "goromonzi": (-17.867, 31.383),
    "ruwa": (-17.883, 31.233), "redcliff": (-19.033, 29.783), "shurugwi": (-19.667, 30.0),
    "guruve": (-16.65, 30.667), "mount darwin": (-16.767, 31.583), "centenary": (-16.817, 31.117),
    "mazowe": (-17.517, 30.967), "glendale": (-17.35, 31.067), "concession": (-17.383, 30.967),
    "mashonaland central": (-16.5, 30.5), "mashonaland east": (-18.0, 31.5),
    "mashonaland west": (-17.5, 29.5), "manicaland": (-18.5, 32.5),
    "matabeleland north": (-18.5, 27.5), "matabeleland south": (-21.0, 29.5),
    "midlands": (-19.0, 29.5), "masvingo province": (-20.0, 31.0),
}

app = FastAPI(title="Farm Advisory API")

def _parse_origins(value: str | None) -> list[str]:
    if not value:
        return ["*"]
    origins = [origin.strip() for origin in value.split(",") if origin.strip()]
    return origins or ["*"]

cors_origins = _parse_origins(os.getenv("CORS_ORIGINS"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resources at startup
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA_FILE = os.path.join(ROOT, "unified_agriculture_data_production.csv")
if not os.path.exists(DATA_FILE):
    DATA_FILE = os.path.join(ROOT, "unified_agriculture_data_real.csv")
if not os.path.exists(DATA_FILE):
    DATA_FILE = os.path.join(ROOT, "unified_agriculture_data.csv")
MODEL_FILE = os.path.join(ROOT, "crop_suitability_classifier_production.pkl")
if not os.path.exists(MODEL_FILE):
    MODEL_FILE = os.path.join(ROOT, "best_classifier_random_forest.pkl")
ALT_MODEL_FILE = os.path.join(ROOT, "best_classifier_random_forest.pkl")
SCALER_FILE = os.path.join(ROOT, "feature_scaler_production.pkl")
META_FILE = os.path.join(ROOT, "model_metadata.json")

df = None
model = None
scaler = None
rule_engine = None
feature_cols = None


def load_resources():
    import sys
    sys.path.insert(0, ROOT)
    from agritex_rules import AgritexRuleEngine

    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    rule_engine = AgritexRuleEngine()

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None and os.path.exists(META_FILE):
        with open(META_FILE) as f:
            model_features = json.load(f).get("selected_features")

    for col in ["rainfall_max", "rainfall_q25", "rainfall_q75", "rainfall_cv"]:
        if col not in df.columns and "rainfall_mean" in df.columns:
            std = df.get("rainfall_std", pd.Series([50] * len(df)))
            if col == "rainfall_max":
                df[col] = df["rainfall_mean"] + 2 * std
            elif col == "rainfall_q25":
                df[col] = df["rainfall_mean"] - std
            elif col == "rainfall_q75":
                df[col] = df["rainfall_mean"] + std
            elif col == "rainfall_cv":
                df[col] = np.where(df["rainfall_mean"] > 0, std / df["rainfall_mean"], 0.3)

    expected = [
        "bulk_density", "coarse_fragments", "nitrogen", "organic_carbon_density",
        "sand", "soil_clay", "soil_organic_carbon_stock", "soil_ph", "soil_soc",
        "rainfall_mean", "rainfall_std", "rainfall_max", "rainfall_q25", "rainfall_q75", "rainfall_cv"
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = 0 if "rainfall" in col else 6
    if model_features:
        for col in model_features:
            if col not in df.columns:
                df[col] = 0
        feature_cols = list(model_features)
    else:
        feature_cols = [c for c in expected if c in df.columns]

    # Keep model/features compatible. If the primary model expects a different
    # number of features than the selected feature list, try a fallback model.
    model_n_features = getattr(model, "n_features_in_", None)
    if model_n_features is not None and model_n_features != len(feature_cols):
        if os.path.exists(ALT_MODEL_FILE):
            try:
                alt_model = joblib.load(ALT_MODEL_FILE)
                alt_n_features = getattr(alt_model, "n_features_in_", None)
                if alt_n_features == len(feature_cols):
                    model = alt_model
                    model_n_features = alt_n_features
            except Exception:
                pass

    # Final safeguard: if still mismatched, adapt feature list length to model.
    if model_n_features is not None and model_n_features != len(feature_cols):
        candidates = [c for c in expected if c in df.columns]
        if len(candidates) >= model_n_features:
            feature_cols = candidates[:model_n_features]

    X_arr = df[feature_cols].fillna(0).values
    if os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_arr)

    # Precompute suitability for dashboard.
    # If the saved scaler was fit on a different feature count (e.g., old model artifacts),
    # refit it on the current feature set so startup never crashes.
    try:
        X_scaled = scaler.transform(X_arr)
    except ValueError:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_arr)
        X_scaled = scaler.transform(X_arr)
        try:
            joblib.dump(scaler, SCALER_FILE)
        except Exception:
            pass
    df["suitability_score"] = model.predict_proba(X_scaled)[:, 1] * 100

    return df, model, scaler, rule_engine, feature_cols


import sys
sys.path.insert(0, ROOT)


@app.on_event("startup")
def startup():
    global df, model, scaler, rule_engine, feature_cols
    df, model, scaler, rule_engine, feature_cols = load_resources()


class AnalyzeRequest(BaseModel):
    lat: float | None = None
    lon: float | None = None
    place: str | None = None
    # Manual soil/rainfall sample fields (all optional)
    soil_ph: float | None = None
    nitrogen: float | None = None
    organic_carbon_density: float | None = None
    sand: float | None = None
    soil_clay: float | None = None
    rainfall_mean: float | None = None
    rainfall_std: float | None = None


class PlaceSearchRequest(BaseModel):
    q: str


@app.get("/api/health")
def health():
    return {"status": "ok", "data_points": len(df) if df is not None else 0}


@app.get("/api/stats")
def stats():
    if df is None:
        raise HTTPException(500, "Data not loaded")
    suitable = (df["suitability_score"] > 50).sum()
    avg_rain = float(df["rainfall_mean"].mean())
    if avg_rain < 150:  # likely monthly mm, convert to annual
        avg_rain *= 12
    avg_ph = float(df["soil_ph"].mean())
    if avg_ph > 14:  # SoilGrids can use pH×100; convert to 0–14 scale
        avg_ph = avg_ph / 100
    return {
        "total_locations": len(df),
        "suitable_sites": int(suitable),
        "suitable_pct": round(100 * suitable / len(df), 1),
        "avg_rainfall": round(avg_rain, 0),
        "avg_ph": round(avg_ph, 2),
    }


@app.post("/api/places/search")
def search_places(req: PlaceSearchRequest):
    q = (req.q or "").strip().lower()
    if not q:
        return {"places": []}
    results = []
    for name, (lat, lon) in ZIMBABWE_PLACES.items():
        if q in name:
            results.append({"name": name.title(), "lat": lat, "lon": lon})
    return {"places": results[:10]}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    global df, model, scaler, rule_engine, feature_cols
    if df is None or model is None:
        raise HTTPException(500, "Model not loaded")

    # If any manual sample fields are provided, use them for analysis
    manual_fields = [
        'soil_ph', 'nitrogen', 'organic_carbon_density', 'sand', 'soil_clay', 'rainfall_mean', 'rainfall_std'
    ]
    manual_sample = {k: getattr(req, k) for k in manual_fields if getattr(req, k) is not None}
    if manual_sample:
        # Fill missing fields with 0
        for k in manual_fields:
            if k not in manual_sample:
                manual_sample[k] = 0.0
        ctx = manual_sample.copy()
        # Derived/optional fields
        ctx['rainfall_cv'] = ctx['rainfall_std'] / ctx['rainfall_mean'] if ctx['rainfall_mean'] else 0.3
        for k in ['rainfall_max', 'rainfall_q25', 'rainfall_q75']:
            ctx[k] = 0.0
        # ML model expects feature_cols order
        features = [ctx.get(col, 0.0) for col in feature_cols]
        X_scaled = scaler.transform(np.array(features).reshape(1, -1))
        prob = float(model.predict_proba(X_scaled)[0, 1])
        rule_results = rule_engine.evaluate(ctx)
        rain = ctx['rainfall_mean']
        zone = "High rainfall" if rain > 1000 else "Good rainfall" if rain > 750 else "Moderate" if rain > 650 else "Low rainfall" if rain > 450 else "Very low rainfall"
        crops = []
        seen = set()
        crop_rules = {
            "Maize": ["maize", "sc700", "sc300", "sc400"],
            "Sorghum": ["sorghum", "red sorghum", "brewing"],
            "Pearl Millet": ["pearl millet", "mhunga"],
            "Finger Millet": ["finger millet", "rapoko"],
            "Tobacco": ["tobacco", "tobacco belt"],
            "Cotton": ["cotton"],
            "Groundnuts": ["groundnut", "pegging", "gypsum"],
            "Sunflower": ["sunflower"],
            "Soybeans": ["soybean", "legume", "rhizobium"],
            "Horticulture": ["potatoes", "tomatoes", "cabbages", "horticulture"],
            "Livestock/Fodder": ["fodder", "lablab", "velvet bean", "pastures", "dairy"],
        }
        for r in rule_results.get("triggered_rules", []):
            if r.get("id") in (19, 43, 51):
                continue
            txt = (r.get("desc", "") + " " + r.get("rec", "")).lower()
            for crop, keys in crop_rules.items():
                if crop not in seen and any(k in txt for k in keys):
                    crops.append(crop)
                    seen.add(crop)
        return {
            "lat": None,
            "lon": None,
            "suitability_pct": round(prob * 100, 1),
            "zone": zone,
            "rainfall_mm": round(rain, 0),
            "soil_ph": round(float(ctx.get("soil_ph", 0)), 1),
            "suggested_crops": crops,
            "recommendations": rule_results.get("recommendations", []),
            "categories": rule_results.get("categories", {}),
            "model": "Random Forest (manual sample)",
        }

    # Otherwise, use location/place as before
    lat, lon = req.lat, req.lon
    if req.place:
        place_str = str(req.place).strip() if req.place else ""
        if not place_str:
            raise HTTPException(400, "Place name is empty")
        q = place_str.lower()
        q_clean = q.replace(" ", "")
        found = False
        for name, coords in ZIMBABWE_PLACES.items():
            n_clean = name.replace(" ", "")
            if q_clean == n_clean or q in name or name in q or q_clean in n_clean or n_clean in q_clean:
                lat, lon = coords
                found = True
                break
        if not found:
            # Geocode via Nominatim - try multiple query formats
            try:
                from geopy.geocoders import Nominatim
                geo = Nominatim(user_agent="farm-advisory-zimbabwe-v1")
                for query in [f"{place_str}, Zimbabwe", place_str, f"{place_str}, ZW"]:
                    loc = geo.geocode(query, timeout=8, country_codes="zw")
                    if loc:
                        lat, lon = loc.latitude, loc.longitude
                        if -23 <= lat <= -15 and 25 <= lon <= 34:
                            found = True
                            break
            except Exception:
                pass
        if not found:
            raise HTTPException(404, f"Place '{place_str}' not found. Try: Harare, Bulawayo, Gweru, Mutare, or any Zimbabwe town.")

    if lat is None or lon is None:
        raise HTTPException(400, "Provide lat/lon or place name")

    dist = (df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2
    idx = dist.idxmin()
    row = df.loc[idx]

    ctx = row.to_dict()

    # CHIRPS data is typically monthly mm; AGRITEX rules expect annual mm
    raw_rain = float(row.get("rainfall_mean", 0))
    rain_annual = raw_rain * 12 if raw_rain < 150 else raw_rain  # convert monthly -> annual if needed
    raw_std = float(row.get("rainfall_std", 50))
    std_annual = raw_std * 12 if raw_rain < 150 else raw_std
    ctx["rainfall_mean"] = rain_annual
    ctx["rainfall_std"] = std_annual
    if "rainfall_cv" not in ctx or pd.isna(ctx.get("rainfall_cv")):
        ctx["rainfall_cv"] = std_annual / rain_annual if rain_annual > 0 else 0.3
    for k in ["rainfall_max", "rainfall_q25", "rainfall_q75"]:
        if k in ctx and pd.notna(ctx.get(k)) and raw_rain < 150:
            ctx[k] = ctx[k] * 12

    # ML model uses raw features (trained on that scale)
    features = row[feature_cols].fillna(0).values.astype(np.float64).reshape(1, -1)
    X_scaled = scaler.transform(features)
    prob = float(model.predict_proba(X_scaled)[0, 1])
    rule_results = rule_engine.evaluate(ctx)

    rain = rain_annual
    zone = "High rainfall" if rain > 1000 else "Good rainfall" if rain > 750 else "Moderate" if rain > 650 else "Low rainfall" if rain > 450 else "Very low rainfall"

    # Extract crops only from rules that explicitly recommend them (not generic advice)
    crops = []
    seen = set()
    crop_rules = {
        "Maize": ["maize", "sc700", "sc300", "sc400"],
        "Sorghum": ["sorghum", "red sorghum", "brewing"],
        "Pearl Millet": ["pearl millet", "mhunga"],
        "Finger Millet": ["finger millet", "rapoko"],
        "Tobacco": ["tobacco", "tobacco belt"],
        "Cotton": ["cotton"],
        "Groundnuts": ["groundnut", "pegging", "gypsum"],
        "Sunflower": ["sunflower"],
        "Soybeans": ["soybean", "legume", "rhizobium"],
        "Horticulture": ["potatoes", "tomatoes", "cabbages", "horticulture"],
        "Livestock/Fodder": ["fodder", "lablab", "velvet bean", "pastures", "dairy"],
    }
    for r in rule_results.get("triggered_rules", []):
        if r.get("id") in (19, 43, 51):
            continue
        txt = (r.get("desc", "") + " " + r.get("rec", "")).lower()
        for crop, keys in crop_rules.items():
            if crop not in seen and any(k in txt for k in keys):
                crops.append(crop)
                seen.add(crop)

    return {
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "suitability_pct": round(prob * 100, 1),
        "zone": zone,
        "rainfall_mm": round(rain, 0),
        "soil_ph": round(float(row.get("soil_ph", 0)), 1),
        "suggested_crops": crops,
        "recommendations": rule_results.get("recommendations", []),
        "categories": rule_results.get("categories", {}),
        "model": "Random Forest (trained on your data)",
    }
