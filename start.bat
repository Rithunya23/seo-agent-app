@echo off
echo Starting SEO Agent...
call venv\Scripts\activate
start "" "index.html"
uvicorn backend:app --reload --port 8000