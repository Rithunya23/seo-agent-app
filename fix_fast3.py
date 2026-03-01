import re

print("=== SEO Agent Speed Fix ===\n")

with open('backend.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

original = content
changes = 0

# 1. Fix TCPConnector limit - match any number
m = re.search(r'TCPConnector\(limit=(\d+)', content)
if m and int(m.group(1)) < 30:
    content = re.sub(r'TCPConnector\(limit=\d+', 'TCPConnector(limit=30', content)
    changes += 1
    print(f"[1] Connections: {m.group(1)} -> 30")
else:
    print(f"[1] Connections: already {m.group(1) if m else 'not found'}")

# 2. Fix timeout - match any number
m = re.search(r'ClientTimeout\(total=(\d+)', content)
if m and int(m.group(1)) > 10:
    content = re.sub(r'ClientTimeout\(total=\d+', 'ClientTimeout(total=8', content)
    changes += 1
    print(f"[2] Timeout: {m.group(1)}s -> 8s")
else:
    print(f"[2] Timeout: already {m.group(1) if m else 'not found'}s")

# 3. Fix all wait_for_timeout - reduce any value over 500
for m in re.finditer(r'wait_for_timeout\((\d+)\)', content):
    val = int(m.group(1))
    if val > 500:
        content = content.replace(f'wait_for_timeout({val})', 'wait_for_timeout(500)', 1)
        changes += 1
        print(f"[3] Wait: {val}ms -> 500ms")

# 4. Fix all asyncio.sleep - reduce any value over 0.3
for m in re.finditer(r'asyncio\.sleep\(([\d.]+)\)', content):
    val = float(m.group(1))
    if val > 0.3:
        content = content.replace(f'asyncio.sleep({m.group(1)})', 'asyncio.sleep(0.2)', 1)
        changes += 1
        print(f"[4] Sleep: {m.group(1)}s -> 0.2s")

# 5. Skip LLM for pages with good meta
if 'async def fix_with_llm' in content and 'SPEED FIX' not in content:
    old = 'async def fix_with_llm(self, title, description, content, url):'
    new = '''async def fix_with_llm(self, title, description, content, url):
        # SPEED FIX: Skip if meta already good
        if title and 10 < len(title.strip()) < 65 and description and 40 < len(description.strip()) < 165:
            return title.strip(), description.strip()'''
    if old in content:
        content = content.replace(old, new)
        changes += 1
        print("[5] Added LLM skip for good meta tags")
    else:
        print("[5] fix_with_llm signature not matched")
else:
    print("[5] LLM skip already applied or function not found")

if content != original:
    with open('backend.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n SUCCESS! {changes} optimizations applied.")
    print(" Restart: uvicorn backend:app --reload --port 8000")
else:
    print("\n No changes needed.")