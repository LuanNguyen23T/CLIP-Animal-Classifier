from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "CLIP Pet Classifier"
    API_V1_STR: str = "/api/v1"
    
    POSTGRES_USER: str = "admin"
    POSTGRES_PASSWORD: str = "secret"
    POSTGRES_DB: str = "pet_classifier"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: str = "5432"

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()
