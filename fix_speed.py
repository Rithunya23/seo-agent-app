with open('backend.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# 1. Increase concurrent connections from 10 to 30
content = content.replace('TCPConnector(limit=10', 'TCPConnector(limit=30')

# 2. Reduce timeout from 15 to 8 seconds
content = content.replace('total=15, connect=10', 'total=8, connect=5')

# 3. Reduce Playwright wait from 2000ms to 500ms
content = content.replace('wait_for_timeout(2000)', 'wait_for_timeout(500)')
content = content.replace('wait_for_timeout(3000)', 'wait_for_timeout(500)')
content = content.replace('wait_for_timeout(1000)', 'wait_for_timeout(300)')

# 4. Skip LLM calls if no API key (use fast rule-based fixes)
# Already handled in code

with open('backend.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('SUCCESS! Backend optimized for speed.')