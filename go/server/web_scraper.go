package server

import (
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

// WebScraper fetches and parses web pages for RAG ingestion. OPT-73.
type WebScraper struct {
	client    *http.Client
	maxDepth  int
	maxPages  int
	userAgent string
}

// NewWebScraper creates a web scraper.
func NewWebScraper(maxDepth, maxPages int) *WebScraper {
	return &WebScraper{
		client:    &http.Client{Timeout: 30 * time.Second},
		maxDepth:  maxDepth,
		maxPages:  maxPages,
		userAgent: "infergo-scraper/1.0",
	}
}

// ScrapedPage holds the extracted content from a web page.
type ScrapedPage struct {
	URL   string
	Title string
	Text  string
	Links []string
}

// Fetch downloads and parses a web page, returning clean text.
func (ws *WebScraper) Fetch(url string) (*ScrapedPage, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", ws.userAgent)

	resp, err := ws.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024)) // 10MB max
	if err != nil {
		return nil, err
	}

	html := string(body)
	page := &ScrapedPage{URL: url}

	// Extract title
	titleRe := regexp.MustCompile(`<title[^>]*>(.*?)</title>`)
	if m := titleRe.FindStringSubmatch(html); len(m) > 1 {
		page.Title = strings.TrimSpace(m[1])
	}

	// Extract links
	linkRe := regexp.MustCompile(`href="(https?://[^"]+)"`)
	for _, m := range linkRe.FindAllStringSubmatch(html, -1) {
		if len(m) > 1 {
			page.Links = append(page.Links, m[1])
		}
	}

	// Strip HTML to text
	page.Text = stripHTMLTags(html)

	return page, nil
}

// Crawl recursively fetches pages up to maxDepth.
func (ws *WebScraper) Crawl(startURL string) []ScrapedPage {
	visited := make(map[string]bool)
	var pages []ScrapedPage

	ws.crawlRecursive(startURL, 0, visited, &pages)
	return pages
}

func (ws *WebScraper) crawlRecursive(url string, depth int, visited map[string]bool, pages *[]ScrapedPage) {
	if depth > ws.maxDepth || len(*pages) >= ws.maxPages || visited[url] {
		return
	}
	visited[url] = true

	page, err := ws.Fetch(url)
	if err != nil {
		return
	}
	*pages = append(*pages, *page)

	if depth < ws.maxDepth {
		for _, link := range page.Links {
			if !visited[link] && len(*pages) < ws.maxPages {
				ws.crawlRecursive(link, depth+1, visited, pages)
			}
		}
	}
}

func stripHTMLTags(html string) string {
	// Remove script and style
	scriptRe := regexp.MustCompile(`(?is)<(script|style)[^>]*>.*?</\1>`)
	html = scriptRe.ReplaceAllString(html, "")

	// Remove tags
	tagRe := regexp.MustCompile(`<[^>]+>`)
	text := tagRe.ReplaceAllString(html, " ")

	// Decode entities
	text = strings.ReplaceAll(text, "&amp;", "&")
	text = strings.ReplaceAll(text, "&lt;", "<")
	text = strings.ReplaceAll(text, "&gt;", ">")
	text = strings.ReplaceAll(text, "&nbsp;", " ")
	text = strings.ReplaceAll(text, "&quot;", "\"")

	// Collapse whitespace
	spaceRe := regexp.MustCompile(`\s+`)
	text = spaceRe.ReplaceAllString(text, " ")

	return strings.TrimSpace(text)
}
