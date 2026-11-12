from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    public_base_url: str = "http://localhost:8787"
    stac_api_url: str = "https://earth-search.aws.element84.com/v1/search"
    stac_collection: str = "sentinel-2-l2a"
    sentinel_hub_client_id: str | None = None
    sentinel_hub_client_secret: str | None = None
    sentinel_hub_token_url: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    sentinel_hub_process_url: str = "https://sh.dataspace.copernicus.eu/api/v1/process"
    vlm_provider: str = "none"
    ollama_base_url: str = "http://localhost:11434"
    ollama_vlm_model: str = "qwen2.5vl:7b"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
