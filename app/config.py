from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = Field(validation_alias="DATABASE_URL")
    openai_api_key: str = Field(validation_alias="OPENAI_API_KEY")
    openai_embed_model: str = Field(default="text-embedding-3-small", validation_alias="OPENAI_EMBED_MODEL")
    openai_chat_model: str = Field(default="gpt-4.1-mini", validation_alias="OPENAI_CHAT_MODEL")
    top_k: int = Field(default=6, validation_alias="TOP_K")
    basic_auth_username: str = Field(default="admin", validation_alias="BASIC_AUTH_USERNAME")
    basic_auth_password: str = Field(default="admin123", validation_alias="BASIC_AUTH_PASSWORD")
    swagger_title: str = Field(default="Agentic RAG API", validation_alias="SWAGGER_TITLE")
    swagger_description: str = Field(
        default="Protected API documentation for the Agentic RAG service.",
        validation_alias="SWAGGER_DESCRIPTION",
    )
    swagger_version: str = Field(default="1.0.0", validation_alias="SWAGGER_VERSION")

settings = Settings()
