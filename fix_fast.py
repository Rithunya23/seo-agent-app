import re

with open('backend.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

changes = 0

# 1. Find the fix_with_llm function and add a skip check
# Look for where it calls OpenAI for each page and add early return if meta is already good
old_fix_pattern = 'async def fix_with_llm(self, title, description, content, url):'
if old_fix_pattern in content:
    new_fix = '''async def fix_with_llm(self, title, description, content, url):
        # SPEED FIX: Skip LLM call if title AND description already exist and are good length
        title_ok = title and 10 < len(title) < 60
        desc_ok = description and 50 < len(description) < 160
        if title_ok and desc_ok:
            return title, description'''
    content = content.replace(old_fix_pattern, new_fix)
    changes += 1
    print("  [1] Added skip check for pages with good meta tags")

# 2. Reduce Playwright wait times
for old, new in [
    ('wait_for_timeout(2000)', 'wait_for_timeout(500)'),
    ('wait_for_timeout(3000)', 'wait_for_timeout(500)'),
    ('wait_for_timeout(1000)', 'wait_for_timeout(300)'),
    ('wait_for_timeout(5000)', 'wait_for_timeout(500)'),
]:
    if old in content:
        content = content.replace(old, new)
        changes += 1
        print(f"  [2] Reduced wait: {old} -> {new}")

# 3. Reduce crawl delay between pages
for old, new in [
    ('asyncio.sleep(2)', 'asyncio.sleep(0.5)'),
    ('asyncio.sleep(1.5)', 'asyncio.sleep(0.3)'),
    ('asyncio.sleep(1)', 'asyncio.sleep(0.3)'),
    ('asyncio.sleep(3)', 'asyncio.sleep(0.5)'),
]:
    if old in content:
        content = content.replace(old, new)
        changes += 1
        print(f"  [3] Reduced sleep: {old} -> {new}")

# 4. Increase concurrent connections
if 'TCPConnector(limit=10' in content:
    content = content.replace('TCPConnector(limit=10', 'TCPConnector(limit=30')
    changes += 1
    print("  [4] Increased concurrent connections: 10 -> 30")
elif 'TCPConnector(limit=30' in content:
    print("  [4] Already at 30 connections")

# 5. Reduce aiohttp timeout
if 'total=15' in content:
    content = content.replace('total=15', 'total=8')
    changes += 1
    print("  [5] Reduced aiohttp timeout: 15s -> 8s")
elif 'total=8' in content:
    print("  [5] Already at 8s timeout")

with open('backend.py', 'w', encoding='utf-8') as f:
    f.write(content)

if changes > 0:
    print(f"\nSUCCESS! Applied {changes} speed optimizations.")
else:
    print("\nNo changes needed or patterns not found. Backend may already be optimized.")
print("Restart the server to apply changes.")