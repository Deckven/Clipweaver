"""Tests never read your own .env: clipmaker.config would otherwise pick up your
catalog, projects and footage folders. Set before any test imports the package."""
import os

os.environ["CLIPMAKER_DOTENV"] = os.devnull
