from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DatabaseServerConfig(BaseModel):
    class Config:
        allow_population_by_field_name = True

    url: Optional[str] = None
    hostname: Optional[str] = "localhost"
    port: int = 5432
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    type: Optional[str] = "postgresql"
    schema_name: Optional[str] = Field(default="public", alias="schema")
    title: Optional[str] = "Untitled"


class ChatDatabaseServerConfig(BaseModel):
    class Config:
        allow_population_by_field_name = True

    hostname: Optional[str] = "localhost"
    port: int = 5432
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    schema_name: Optional[str] = Field(default="public", alias="schema")


class ConfigPayload(BaseModel):
    database_server: DatabaseServerConfig
    chat_database_server: ChatDatabaseServerConfig


class ConnectRequest(ConfigPayload):
    save: bool = True
    create_chat_database_if_missing: bool = False
