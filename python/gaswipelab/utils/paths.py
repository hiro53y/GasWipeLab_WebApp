"""paths.py — Webアプリ用スタブ（ファイルシステムアクセスなし）"""
from __future__ import annotations
from pathlib import Path


def app_root() -> Path:
    return Path("/")


def config_dir() -> Path:
    return Path("/configs")


def data_dir() -> Path:
    return Path("/data")


def docs_dir() -> Path:
    return Path("/docs")


def user_data_dir() -> Path:
    return Path("/user_data")


def projects_dir() -> Path:
    return Path("/user_data/projects")


def exports_dir() -> Path:
    return Path("/exports")


def logs_dir() -> Path:
    return Path("/logs")
