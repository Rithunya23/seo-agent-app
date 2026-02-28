with open('backend.py', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'await self._crawl_playwright()' in line:
        new_lines.append(line.replace('await self._crawl_playwright()', 'await self._crawl_httpx_fallback()'))
    else:
        new_lines.append(line)

with open('backend.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('SUCCESS! Playwright replaced with httpx for Render compatibility.')