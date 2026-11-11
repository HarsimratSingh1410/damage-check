"""Configuration and settings for the Damage-Check Module.

Reads from environment variables (prefix DAMAGE_) and an optional .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Stage selection
    segmenter: str = "rembg"    # "rembg" | "sam"
    aligner: str = "sift"       # "sift" | "loftr"
    comparator: str = "ssim"    # "ssim" | "dino"
    enable_vlm: bool = False

    # Quality-gate thresholds
    blur_threshold: float = 100.0       # Laplacian variance
    brightness_min: float = 30.0
    brightness_max: float = 220.0
    min_object_fraction: float = 0.05   # object pixels / total pixels
    alignment_min_score: float = 0.3

    # Damage detection thresholds
    damage_score_threshold: float = 0.4
    min_region_pixels: int = 200

    # Storage / output
    storage_backend: str = "local"      # "local" | "s3"
    output_dir: str = "output"

    # Device
    device: str = "cpu"                 # "cpu" | "cuda"

    model_config = {
        "env_prefix": "DAMAGE_",
        "env_file": ".env",
        "extra": "ignore",
    }


settings = Settings()
