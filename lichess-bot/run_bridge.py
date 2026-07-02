#!/usr/bin/env python3
"""Bootstrap script for the lichess-bot bridge running MyEngine.jar.

Expects lichess-bot's own source (lichess-bot.py, requirements.txt, ...) from
https://github.com/lichess-bot-devs/lichess-bot to be cloned/copied into this
same directory, alongside config.yml and ./engines/MyEngine.jar.

Usage:
    python run_bridge.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yml"
BOT_ENTRY = BASE_DIR / "lichess-bot.py"
REQUIREMENTS = BASE_DIR / "requirements.txt"
ENGINE_JAR = BASE_DIR / "engines" / "MyEngine.jar"

MIN_PYTHON = (3, 9)


def check_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        sys.exit(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required, "
            f"found {sys.version_info.major}.{sys.version_info.minor}."
        )


def check_java_available() -> None:
    if shutil.which("java") is None:
        sys.exit(
            "No 'java' executable found on PATH. The engine is launched via "
            "'java -jar MyEngine.jar' — install a JRE/JDK and ensure it's on PATH."
        )


def check_config_exists() -> None:
    if not CONFIG_PATH.is_file():
        sys.exit(
            f"Missing config file: {CONFIG_PATH}\n"
            "Create/edit config.yml (with your real Lichess token) before starting."
        )


def check_engine_jar_exists() -> None:
    if not ENGINE_JAR.is_file():
        print(f"Warning: {ENGINE_JAR} not found yet — build/copy MyEngine.jar there before playing.")


def check_bot_entry_exists() -> None:
    if not BOT_ENTRY.is_file():
        sys.exit(
            f"Missing {BOT_ENTRY.name} in {BASE_DIR}.\n"
            "Clone the lichess-bot framework into this directory first, e.g.:\n"
            "  git clone https://github.com/lichess-bot-devs/lichess-bot ."
        )


def check_dependencies() -> None:
    if not REQUIREMENTS.is_file():
        print(f"Warning: {REQUIREMENTS.name} not found, skipping dependency check.")
        return
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        print(f"Warning: could not run 'pip check' ({exc}); continuing anyway.")
        return
    if result.returncode != 0:
        print("Warning: 'pip check' reported dependency issues:")
        print((result.stdout or result.stderr).strip())
        print(f"Run: {sys.executable} -m pip install -r {REQUIREMENTS.name}")


def start_bridge() -> None:
    print(f"Starting lichess-bot from {BASE_DIR} ...")
    try:
        subprocess.run([sys.executable, str(BOT_ENTRY)], cwd=BASE_DIR, check=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(f"lichess-bot exited with error code {exc.returncode}.")
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")


def main() -> None:
    check_python_version()
    check_java_available()
    check_config_exists()
    check_engine_jar_exists()
    check_bot_entry_exists()
    check_dependencies()
    start_bridge()


if __name__ == "__main__":
    main()
