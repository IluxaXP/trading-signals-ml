from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import joblib


REQUIRED_KEYS = ("model", "scaler", "features")


def validate_bundle(bundle: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in bundle]
    if missing:
        raise ValueError(f"Model bundle is invalid. Missing keys: {missing}")
    if not isinstance(bundle["features"], list) or not bundle["features"]:
        raise ValueError("Model bundle key 'features' must be a non-empty list")


@lru_cache(maxsize=2)
def load_model_bundle(model_path: str) -> dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise TypeError("Model artifact must be a dict bundle")
    validate_bundle(bundle)
    return bundle
