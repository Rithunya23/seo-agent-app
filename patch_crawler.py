"""
Patch script — Adds Playwright headless browser to the Crawler in backend.py
Run: python patch_crawler.py
"""

# Read the current backend.py
with open('backend.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Check if already patched
if '_crawl_playwright' in content:
    print("Already patched! Playwright crawler is present.")
    exit()

# Find the old Crawler class and replace it
OLD_MARKER = "class Crawler:"
if OLD_MARKER not in content:
    print("ERROR: Could not find 'class Crawler:' in backend.py")
    exit()

# Find where Crawler class starts and where SEO ANALYZER starts
crawler_start = content.index(OLD_MARKER)
# Go back to find the comment above it
comment_start = content.rfind("# ", 0, crawler_start)
# Find where SEO ANALYZER section starts
analyzer_marker = "# SEO ANALYZER" if "# SEO ANALYZER" in content else "class SEOAnalyzer:"
analyzer_start = content.index(analyzer_marker)
# Go back to find the comment line
if "# SEO ANALYZER" in content:
    section_start = content.rfind("# ", 0, analyzer_start)
else:
    section_start = analyzer_start

NEW_CRAWLER = '''# ===========================================================================
# WEB CRAWLER (Hybrid - Playwright for bot-protected sites + aiohttp fallback)
# ===========================================================================
class Crawler:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }

    def __init__(self, base_url, max_pages=100):
        self.base_url = base_url if base_url.startswith("http") else f"https://{base_url}"
        self.max_pages = max_pages
        self.visited = set()
        self.results = []
        self.use_playwright = False
        p = urlparse(self.base_url)
        self.domain = p.netloc
        self.scheme = p.scheme

    async def crawl(self):
        import re as _re
        # First try aiohttp (fast)
        conn = aiohttp.TCPConnector(limit=10, ssl=False)
        timeout = aiohttp.ClientTimeout(total=15, connect=10)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout, headers=self.HEADERS) as s:
            sitemap_urls = await self._try_sitemap(s)
            logger.info(f"Sitemap found {len(sitemap_urls)} URLs")
            await self._crawl_aiohttp(s, self.base_url)
            # If blocked, switch to Playwright
            if len(self.results) == 0 or (len(self.results) == 1 and self.results[0].get("error")):
                logger.info("aiohttp blocked - switching to Playwright")
                self.visited.clear()
                self.results.clear()
                self.use_playwright = True
                await self._crawl_playwright()
            else:
                for url in sitemap_urls:
                    if len(self.visited) >= self.max_pages:
                        break
                    if url not in self.visited:
                        await self._crawl_aiohttp(s, url)
        if len(self.results) == 0 and not self.use_playwright:
            self.use_playwright = True
            await self._crawl_playwright()
        logger.info(f"Crawl complete: {len(self.results)} pages found")
        return self.results

    async def _crawl_playwright(self):
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.warning("Playwright not installed, using httpx fallback")
            await self._crawl_httpx_fallback()
            return
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    viewport={"width": 1920, "height": 1080}
                )
                page = await context.new_page()
                urls_to_crawl = [self.base_url]
                while urls_to_crawl and len(self.visited) < self.max_pages:
                    url = urls_to_crawl.pop(0)
                    url = url.split('#')[0].split('?')[0].rstrip('/')
                    if not url or url in self.visited:
                        continue
                    self.visited.add(url)
                    try:
                        resp = await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                        if not resp or resp.status != 200:
                            self.results.append({"url": url, "path": urlparse(url).path or "/", "status_code": resp.status if resp else 0, "error": f"HTTP {resp.status if resp else 0}"})
                            continue
                        await page.wait_for_timeout(2000)
                        html = await page.content()
                        soup = BeautifulSoup(html, "lxml")
                        self.results.append(self._extract(url, soup, resp.status))
                        links = self._get_links(url, soup)
                        for link in links:
                            if link not in self.visited and link not in urls_to_crawl:
                                urls_to_crawl.append(link)
                    except Exception as e:
                        self.results.append({"url": url, "path": urlparse(url).path or "/", "status_code": 0, "error": str(e)})
                await browser.close()
        except Exception as e:
            logger.error(f"Playwright error: {e}")
            await self._crawl_httpx_fallback()

    async def _crawl_httpx_fallback(self):
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15, headers=self.HEADERS, verify=False) as client:
                urls_to_crawl = [self.base_url]
                while urls_to_crawl and len(self.visited) < self.max_pages:
                    url = urls_to_crawl.pop(0)
                    url = url.split('#')[0].split('?')[0].rstrip('/')
                    if not url or url in self.visited:
                        continue
                    self.visited.add(url)
                    try:
                        resp = await client.get(url)
                        if resp.status_code != 200:
                            self.results.append({"url": url, "path": urlparse(url).path or "/", "status_code": resp.status_code, "error": f"HTTP {resp.status_code}"})
                            continue
                        ct = resp.headers.get("content-type", "")
                        if "text/html" not in ct:
                            continue
                        soup = BeautifulSoup(resp.text, "lxml")
                        self.results.append(self._extract(url, soup, resp.status_code))
                        links = self._get_links(url, soup)
                        for link in links:
                            if link not in self.visited and link not in urls_to_crawl:
                                urls_to_crawl.append(link)
                    except Exception as e:
                        self.results.append({"url": url, "path": urlparse(url).path or "/", "status_code": 0, "error": str(e)})
        except Exception as e:
            logger.error(f"httpx fallback error: {e}")

    async def _try_sitemap(self, session):
        import re as _re
        urls = []
        for path in ["/sitemap.xml", "/sitemap_index.xml", "/sitemap/"]:
            try:
                async with session.get(self.base_url.rstrip('/') + path, allow_redirects=True) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        locs = _re.findall(r'<loc>(.*?)</loc>', text)
                        for loc in locs:
                            parsed = urlparse(loc)
                            if parsed.netloc == self.domain or not parsed.netloc:
                                urls.append(loc)
                        if urls:
                            break
            except Exception:
                continue
        return urls[:self.max_pages]

    async def _crawl_aiohttp(self, session, url):
        if url in self.visited or len(self.visited) >= self.max_pages:
            return
        url = url.split('#')[0].split('?')[0].rstrip('/')
        if not url or url in self.visited:
            return
        self.visited.add(url)
        try:
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status != 200:
                    self.results.append({"url": url, "path": urlparse(url).path or "/", "status_code": resp.status, "error": f"HTTP {resp.status}"})
                    return
                ct = resp.headers.get("Content-Type", "")
                if "text/html" not in ct and "application/xhtml" not in ct:
                    return
                html = await resp.text()
                soup = BeautifulSoup(html, "lxml")
                self.results.append(self._extract(url, soup, resp.status))
                links = self._get_links(url, soup)
                for link in links:
                    if len(self.visited) >= self.max_pages:
                        break
                    if link not in self.visited:
                        await self._crawl_aiohttp(session, link)
        except asyncio.TimeoutError:
            self.results.append({"url": url, "path": urlparse(url).path or "/", "status_code": 0, "error": "Timeout"})
        except Exception as e:
            self.results.append({"url": url, "path": urlparse(url).path or "/", "status_code": 0, "error": str(e)})

    def _extract(self, url, soup, code):
        title_tag = soup.find("title")
        desc_tag = soup.find("meta", attrs={"name": "description"})
        body = soup.find("body")
        text = body.get_text(" ", strip=True) if body else ""
        imgs = soup.find_all("img")
        links = soup.find_all("a", href=True)
        int_links = sum(1 for l in links if l["href"].startswith("/") or self.domain in l.get("href", ""))
        ext_links = sum(1 for l in links if l["href"].startswith("http") and self.domain not in l.get("href", ""))
        return {
            "url": url, "path": urlparse(url).path or "/", "status_code": code,
            "meta_title": title_tag.get_text(strip=True) if title_tag else "",
            "meta_description": desc_tag.get("content", "") if desc_tag else "",
            "h1_tags": [h.get_text(strip=True) for h in soup.find_all("h1")],
            "word_count": len(text.split()),
            "internal_links": int_links, "external_links": ext_links,
            "images_without_alt": sum(1 for i in imgs if not i.get("alt")),
            "total_images": len(imgs),
            "has_schema": bool(soup.find_all("script", type="application/ld+json")),
            "has_canonical": bool(soup.find("link", rel="canonical")),
            "has_og_title": bool(soup.find("meta", property="og:title")),
            "has_og_description": bool(soup.find("meta", property="og:description")),
            "h2_count": len(soup.find_all("h2")), "h3_count": len(soup.find_all("h3")),
        }

    def _get_links(self, url, soup):
        out = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            full = urljoin(url, href)
            p = urlparse(full)
            if p.netloc == self.domain and p.scheme in ("http", "https"):
                clean = f"{p.scheme}://{p.netloc}{p.path}".rstrip("/")
                if not clean:
                    clean = f"{p.scheme}://{p.netloc}"
                skip_ext = ('.pdf','.jpg','.jpeg','.png','.gif','.svg','.css','.js','.zip','.mp4','.mp3','.ico')
                if not any(clean.lower().endswith(ext) for ext in skip_ext):
                    out.add(clean)
        return list(out)


'''

# Replace the old crawler section
new_content = content[:comment_start] + NEW_CRAWLER + content[section_start:]

# Write the patched file
with open('backend.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("SUCCESS! Crawler patched with Playwright support.")
print("Now run: uvicorn backend:app --reload --port 8000")
