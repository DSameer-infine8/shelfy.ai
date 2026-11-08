import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://localhost:8080,*"

    # LLM keys
    LLM_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""

    # Google Books API Key
    GOOGLE_BOOKS_API_KEY: str = ""

    # Model file path
    YOLO_MODEL_PATH: str = "models/best.pt"

    # Supabase (optional)
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    class Config:
        env_file = (
            Path(__file__).resolve().parent / ".env",
            Path(__file__).resolve().parent.parent / ".env",
            ".env",
        )
        extra = "ignore"

    def get_google_books_api_key(self) -> str:
        """Returns the Google Books API key from settings or environment."""
        key = (
            self.GOOGLE_BOOKS_API_KEY
            or os.environ.get("GOOGLE_BOOKS_API_KEY", "")
        ).strip()
        if not key or "your_" in key.lower() or "<" in key:
            return ""
        return key

    @property
    def effective_port(self) -> int:
        """Respects Cloud Run dynamic $PORT environment variable if set."""
        if "PORT" in os.environ:
            try:
                return int(os.environ["PORT"])
            except ValueError:
                pass
        return self.PORT

    def get_gemini_api_key(self) -> str:
        """Returns the first available Gemini API key from supported environment variables."""
        key = (
            self.LLM_API_KEY
            or self.GEMINI_API_KEY
            or self.GOOGLE_API_KEY
            or os.environ.get("LLM_API_KEY", "")
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        ).strip()
        if not key or "your_" in key.lower() or "<" in key:
            return ""
        return key

    def get_allowed_origins_list(self) -> List[str]:
        if not self.ALLOWED_ORIGINS:
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    def resolve_model_path(self) -> Path:
        """Resolves the YOLO model path relative to backend root or absolute."""
        base_dir = Path(__file__).resolve().parent
        candidate = Path(self.YOLO_MODEL_PATH)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        
        rel_path = base_dir / candidate
        if rel_path.exists():
            return rel_path
        
        # Fallback to backend/models/best.pt
        fallback = base_dir / "models" / "best.pt"
        if fallback.exists():
            return fallback

        # Fallback to raw-assets/best.pt
        raw_assets = base_dir.parent / "raw-assets" / "best.pt"
        if raw_assets.exists():
            return raw_assets

        return rel_path

settings = Settings()
