from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, model_validator


# TODO: Upgrade to latest pydantic version
class RedisDBConfig(BaseModel):
    redis_url: str = Field(..., description="Redis URL")
    collection_name: str = Field("mem0", description="Collection name")
    embedding_model_dims: int = Field(1536, description="Embedding model dimensions")

    @model_validator(mode="before")
    @classmethod
    def handle_dimension_alias(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "dimension" in values:
            values["embedding_model_dims"] = values.pop("dimension")
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        # We allow 'dimension' as an alias in the handle_dimension_alias validator
        # which runs before this one if we are careful about the order or if we handle it here.
        # Actually, in Pydantic V2, multiple validators at the same mode run in the order they are defined.
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields - {"dimension"}
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values

    model_config = ConfigDict(arbitrary_types_allowed=True)
