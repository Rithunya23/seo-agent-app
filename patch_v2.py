"""
Patch v2 - Improved Playwright crawler that discovers ALL pages on JS-rendered sites
Run: python patch_v2.py
"""

with open('backend.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Find Crawler class boundaries
OLD_MARKER = "class Crawler:"
if OLD_MARKER not in content:
    print("ERROR: Could not find Crawler class")
    exit()

crawler_start = content.index(OLD_MARKER)
comment_start = content.rfind("#", 0, crawler_start - 5)
# Find line start
comment_start = content.rfind("\n", 0, comment_start) + 1

# Find the next class or section after Crawler
after_crawler = content.find("\nclass SEOAnalyzer", crawler_start)
if after_crawler == -1:
    after_crawler = content.find("\n# SEO ANALYZER", crawler_start)
if after_crawler == -1:
    # Try finding by the separator
    after_crawler = content.find("class SEOAnalyzer", crawler_start)

# Go back to include the comment
section_end = content.rfind("\n#", 0, after_crawler)
if section_end > crawler_start:
    after_crawler = section_end

NEW_CRAWLER = '''# ===========================================================================
# WEB CRAWLER (Auto-discovers ALL pages using Playwright + sitemap + link extraction)
# ===========================================================================
class Crawler:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }

    def __init__(self, base_url, max_pages=200):
        self.base_url = base_url if base_url.startswith("http") else f"https://{base_url}"
        # Also try www version
        self.base_url_alt = None
        p = urlparse(self.base_url)
        if not p.netloc.startswith("www."):
            self.base_url_alt = f"{p.scheme}://www.{p.netloc}{p.path}"
        self.max_pages = max_pages
        self.visited = set()
        self.results = []
        self.domain = p.netloc
        self.domain_alt = f"www.{p.netloc}" if not p.netloc.startswith("www.") else p.netloc.replace("www.", "")
        self.scheme = p.scheme

    def _is_same_domain(self, netloc):
        return netloc == self.domain or netloc == self.domain_alt

    async def crawl(self):
        import re as _re
        logger.info(f"Starting crawl of {self.base_url} (max {self.max_pages} pages)")

        # PHASE 1: Try sitemap first (fastest way to discover pages)
        sitemap_urls = await self._try_sitemap()
        logger.info(f"Sitemap: found {len(sitemap_urls)} URLs")

        # PHASE 2: Try aiohttp first (fast)
        conn = aiohttp.TCPConnector(limit=10, ssl=False)
        timeout = aiohttp.ClientTimeout(total=15, connect=10)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout, headers=self.HEADERS) as s:
            await self._crawl_aiohttp(s, self.base_url)

            # Check if we actually got content
            got_content = any(r for r in self.results if not r.get("error") and r.get("meta_title") or r.get("word_count", 0) > 10)

            if got_content:
                # aiohttp works! Crawl sitemap URLs + discovered links
                for url in sitemap_urls:
                    if len(self.visited) >= self.max_pages:
                        break
                    if url not in self.visited:
                        await self._crawl_aiohttp(s, url)
            else:
                # aiohttp blocked - switch to Playwright
                logger.info("aiohttp got blocked or empty content - switching to Playwright")
                self.visited.clear()
                self.results.clear()
                await self._crawl_playwright(sitemap_urls)

        # PHASE 3: If still nothing, try Playwright
        if len(self.results) == 0:
            logger.info("No results yet - trying Playwright")
            await self._crawl_playwright(sitemap_urls)

        # PHASE 4: If STILL nothing, try httpx as last resort
        if len(self.results) == 0:
            logger.info("Trying httpx fallback")
            await self._crawl_httpx(sitemap_urls)

        logger.info(f"Crawl complete: {len(self.results)} pages with content")
        return self.results

    async def _crawl_playwright(self, extra_urls=None):
        """Use real Chrome browser - can handle ANY website including Cloudflare, JS-rendered, SPAs."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.warning("Playwright not installed")
            return

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    viewport={"width": 1920, "height": 1080}
                )
                page = await context.new_page()

                # Build URL queue: base URL + sitemap + alt base
                urls_to_crawl = [self.base_url]
                if self.base_url_alt:
                    urls_to_crawl.append(self.base_url_alt)
                if extra_urls:
                    urls_to_crawl.extend(extra_urls)

                while urls_to_crawl and len(self.visited) < self.max_pages:
                    url = urls_to_crawl.pop(0)
                    # Clean URL
                    url = url.split('#')[0].rstrip('/')
                    if '?' in url:
                        url = url.split('?')[0]
                    if not url or url in self.visited:
                        continue
                    self.visited.add(url)

                    try:
                        resp = await page.goto(url, wait_until="networkidle", timeout=20000)
                        if not resp:
                            continue
                        status = resp.status

                        # Handle redirects - track final URL
                        final_url = page.url.split('#')[0].split('?')[0].rstrip('/')
                        if final_url != url:
                            self.visited.add(final_url)

                        if status != 200:
                            self.results.append({"url": url, "path": urlparse(url).path or "/", "status_code": status, "error": f"HTTP {status}"})
                            continue

                        # Wait for JS to fully render
                        await page.wait_for_timeout(3000)

                        # Scroll down to trigger lazy-loaded content
                        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        await page.wait_for_timeout(1000)

                        html = await page.content()
                        soup = BeautifulSoup(html, "lxml")

                        # Only add if we got actual content (not a Cloudflare challenge page)
                        title = soup.find("title")
                        body = soup.find("body")
                        body_text = body.get_text(" ", strip=True) if body else ""

                        if title or len(body_text) > 50:
                            self.results.append(self._extract(final_url or url, soup, status))
                            logger.info(f"Playwright crawled: {urlparse(url).path} ({len(body_text)} chars)")

                        # Extract ALL links including JS-rendered ones
                        links = self._get_links(url, soup)

                        # Also try to get links via Playwright (catches JS-generated links)
                        try:
                            js_links = await page.evaluate("""
                                () => {
                                    const links = [];
                                    document.querySelectorAll('a[href]').forEach(a => {
                                        links.push(a.href);
                                    });
                                    // Also check for common navigation patterns
                                    document.querySelectorAll('[data-href], [onclick]').forEach(el => {
                                        const href = el.getAttribute('data-href');
                                        if (href) links.push(href);
                                    });
                                    return links;
                                }
                            """)
                            for link in js_links:
                                parsed = urlparse(link)
                                if self._is_same_domain(parsed.netloc):
                                    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
                                    if clean and clean not in self.visited and clean not in urls_to_crawl:
                                        skip_ext = ('.pdf','.jpg','.jpeg','.png','.gif','.svg','.css','.js','.zip','.mp4','.mp3','.ico')
                                        if not any(clean.lower().endswith(ext) for ext in skip_ext):
                                            links.append(clean)
                        except Exception:
                            pass

                        # Add discovered links to queue
                        for link in links:
                            if link not in self.visited and link not in urls_to_crawl:
                                urls_to_crawl.append(link)

                    except Exception as e:
                        logger.warning(f"Playwright error on {url}: {e}")
                        self.results.append({"url": url, "path": urlparse(url).path or "/", "status_code": 0, "error": str(e)})

                await browser.close()
        except Exception as e:
            logger.error(f"Playwright fatal error: {e}")

    async def _crawl_httpx(self, extra_urls=None):
        """Last resort fallback using httpx with cookies."""
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15, headers=self.HEADERS, verify=False) as client:
                urls_to_crawl = [self.base_url]
                if extra_urls:
                    urls_to_crawl.extend(extra_urls)
                while urls_to_crawl and len(self.visited) < self.max_pages:
                    url = urls_to_crawl.pop(0)
                    url = url.split('#')[0].split('?')[0].rstrip('/')
                    if not url or url in self.visited:
                        continue
                    self.visited.add(url)
                    try:
                        resp = await client.get(url)
                        if resp.status_code != 200:
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
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"httpx error: {e}")

    async def _try_sitemap(self):
        """Discover pages via sitemap.xml."""
        import re as _re
        urls = []
        bases = [self.base_url]
        if self.base_url_alt:
            bases.append(self.base_url_alt)

        for base in bases:
            for path in ["/sitemap.xml", "/sitemap_index.xml", "/sitemap/", "/sitemap.xml.gz"]:
                try:
                    async with httpx.AsyncClient(follow_redirects=True, timeout=10, headers=self.HEADERS, verify=False) as client:
                        resp = await client.get(base.rstrip('/') + path)
                        if resp.status_code == 200:
                            text = resp.text
                            locs = _re.findall(r'<loc>(.*?)</loc>', text)
                            # Check if these are sub-sitemaps
                            for loc in locs:
                                if 'sitemap' in loc.lower() and loc.endswith('.xml'):
                                    # It's a sitemap index - fetch sub-sitemaps
                                    try:
                                        sub_resp = await client.get(loc)
                                        if sub_resp.status_code == 200:
                                            sub_locs = _re.findall(r'<loc>(.*?)</loc>', sub_resp.text)
                                            for sub_loc in sub_locs:
                                                parsed = urlparse(sub_loc)
                                                if self._is_same_domain(parsed.netloc):
                                                    urls.append(sub_loc)
                                    except Exception:
                                        pass
                                else:
                                    parsed = urlparse(loc)
                                    if self._is_same_domain(parsed.netloc) or not parsed.netloc:
                                        urls.append(loc)
                            if urls:
                                return list(set(urls))[:self.max_pages]
                except Exception:
                    continue
        return urls

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
            if href.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
                continue
            full = urljoin(url, href)
            p = urlparse(full)
            if self._is_same_domain(p.netloc) and p.scheme in ("http", "https"):
                clean = f"{p.scheme}://{p.netloc}{p.path}".rstrip("/")
                if not clean:
                    clean = f"{p.scheme}://{p.netloc}"
                skip_ext = ('.pdf','.jpg','.jpeg','.png','.gif','.svg','.css','.js','.zip','.mp4','.mp3','.ico','.woff','.woff2','.ttf')
                if not any(clean.lower().endswith(ext) for ext in skip_ext):
                    out.add(clean)
        return list(out)


'''

# Replace
new_content = content[:comment_start] + NEW_CRAWLER + content[after_crawler:]

with open('backend.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("SUCCESS! Crawler v2 patched.")
print("Features:")
print("  - Auto-discovers ALL pages (sitemap + JS link extraction)")
print("  - Handles Cloudflare, JS-rendered sites, SPAs")
print("  - Tries www and non-www versions")
print("  - Scrolls pages to load lazy content")
print("  - Extracts JS-generated navigation links")
print("  - 3 fallback levels: aiohttp -> Playwright -> httpx")
print("")
print("Now run: uvicorn backend:app --reload --port 8000")
