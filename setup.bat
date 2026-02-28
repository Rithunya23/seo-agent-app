@echo off
echo ============================================
echo   SEO Agent - Autonomous SEO Optimization
echo   One-Click Setup
echo ============================================
echo.

echo [1/5] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo [2/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [3/5] Installing dependencies...
pip install fastapi uvicorn sqlalchemy aiohttp beautifulsoup4 lxml pydantic python-jose passlib httpx python-multipart bcrypt cryptography playwright --quiet

echo [4/5] Installing Chromium browser for crawler...
playwright install chromium

echo [5/5] Setup complete!
echo.
echo ============================================
echo   To start the app:
echo   1. Run: start.bat
echo   2. Open index.html in your browser
echo ============================================
pause