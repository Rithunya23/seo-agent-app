import re

with open('backend.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

changes = 0

# 1. Increase concurrent connections from 5 to 30
if 'TCPConnector(limit=5' in content:
    content = content.replace('TCPConnector(limit=5', 'TCPConnector(limit=30')
    changes += 1
    print("[1] Increased connections: 5 -> 30")
elif 'TCPConnector(limit=10' in content:
    content = content.replace('TCPConnector(limit=10', 'TCPConnector(limit=30')
    changes += 1
    print("[1] Increased connections: 10 -> 30")
else:
    print("[1] Connections already optimized or not found")

# 2. Reduce aiohttp timeout from 30 to 8
if 'total=30' in content:
    content = content.replace('total=30', 'total=8')
    changes += 1
    print("[2] Reduced timeout: 30s -> 8s")
elif 'total=15' in content:
    content = content.replace('total=15', 'total=8')
    changes += 1
    print("[2] Reduced timeout: 15s -> 8s")
else:
    print("[2] Timeout already optimized or not found")

# 3. Reduce all wait_for_timeout values
for old, new in [
    ('wait_for_timeout(5000)', 'wait_for_timeout(500)'),
    ('wait_for_timeout(3000)', 'wait_for_timeout(500)'),
    ('wait_for_timeout(2000)', 'wait_for_timeout(500)'),
    ('wait_for_timeout(1000)', 'wait_for_timeout(300)'),
]:
    if old in content:
        content = content.replace(old, new)
        changes += 1
        print(f"[3] Reduced wait: {old} -> {new}")

# 4. Reduce all asyncio.sleep values
for old, new in [
    ('asyncio.sleep(3)', 'asyncio.sleep(0.3)'),
    ('asyncio.sleep(2)', 'asyncio.sleep(0.3)'),
    ('asyncio.sleep(1.5)', 'asyncio.sleep(0.2)'),
    ('asyncio.sleep(1)', 'asyncio.sleep(0.2)'),
    ('asyncio.sleep(0.5)', 'asyncio.sleep(0.1)'),
]:
    if old in content:
        content = content.replace(old, new)
        changes += 1
        print(f"[4] Reduced sleep: {old} -> {new}")

# 5. Skip LLM call for pages with good meta tags already
fix_func = 'async def fix_with_llm(self, title, description, content, url):'
if fix_func in content and 'SPEED FIX' not in content:
    new_func = '''async def fix_with_llm(self, title, description, content, url):
        # SPEED FIX: Skip LLM if title AND description already exist and are decent
        title_ok = title and len(title.strip()) > 10 and len(title.strip()) < 65
        desc_ok = description and len(description.strip()) > 40 and len(description.strip()) < 165
        if title_ok and desc_ok:
            return title.strip(), description.strip()'''
    content = content.replace(fix_func, new_func)
    changes += 1
    print("[5] Added skip check for pages with good meta tags")
else:
    print("[5] LLM skip already applied or function not found")

with open('backend.py', 'w', encoding='utf-8') as f:
    f.write(content)

if changes > 0:
    print(f"\nSUCCESS! Applied {changes} speed optimizations.")
    print("Restart server: uvicorn backend:app --reload --port 8000")
else:
    print("\nNo changes applied. Backend may already be optimized.")