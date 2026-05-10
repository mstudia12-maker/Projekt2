from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings

_BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    api_prefix: str = ""
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
        "http://localhost:3030",
        "http://127.0.0.1:3030",
        "http://localhost:3040",
        "http://127.0.0.1:3040",
    ]
    data_dir: str = str(_BACKEND_DIR / "data" / "sessions")
    forecast_horizon_months: int = 6
    min_train_months: int = 12
    random_state: int = 42
    afg_project_root: Optional[str] = None
    afg_feature_csv: str = "AFG_ML_FEATUREONLY_SKU_EOM_STRICT.csv"
    afg_train_csv: str = "AFG_ML_TRAIN_SKU_EOM_STRICT.csv"

    class Config:
        env_file = ".env"


settings = Settings()
