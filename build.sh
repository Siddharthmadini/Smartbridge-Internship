#!/usr/bin/env bash
# Render build script — runs once on each deploy

set -e  # exit on error

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Initializing database (safe — uses CREATE TABLE IF NOT EXISTS)..."
python src/database_init.py

echo "Build complete."
