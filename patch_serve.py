with open('backend.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

if 'serve_frontend' in content:
    print('Already patched!')
else:
    old_text = 'app = FastAPI(title="SEO Agent API", version="1.0.0")'
    new_text = old_text + """

from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>SEO Agent</h1><p>index.html not found</p>")
"""
    content = content.replace(old_text, new_text, 1)
    with open('backend.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS!')
```

Save (`Ctrl + S`), then run:
```
python patch_serve.py
```

Should say **SUCCESS!** Then:
```
set OPENAI_API_KEY=sk-proj-yCqAgHcds-wg85sVD70sVLI5Tqlr_w0Ua3QilWVXBExtxctefILB7A3_yQ6ge2hFH13oqEvR8BJT3BlbkFJQ955M-tVc71Mie-eaKJvzQ8voR3Y9uHSRhpqBc5yX5_4UzKnG9x9qtJRAN1AMglFZIIxtmrBMA
```
```
uvicorn backend:app --reload --port 8000