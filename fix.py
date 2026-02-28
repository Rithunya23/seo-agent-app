import re

with open('backend.py', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if 'app.add_middleware(CORSMiddleware' in line:
        new_lines.append('\n')
        new_lines.append('from fastapi.responses import FileResponse, HTMLResponse\n')
        new_lines.append('from pathlib import Path\n')
        new_lines.append('\n')
        new_lines.append('@app.get("/", response_class=HTMLResponse)\n')
        new_lines.append('async def serve_frontend():\n')
        new_lines.append('    html_path = Path(__file__).parent / "index.html"\n')
        new_lines.append('    if html_path.exists():\n')
        new_lines.append('        return FileResponse(html_path)\n')
        new_lines.append('    return HTMLResponse("index.html not found")\n')
        new_lines.append('\n')

with open('backend.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('SUCCESS!' if 'serve_frontend' in ''.join(new_lines) else 'FAILED')