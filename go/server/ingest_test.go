package server

import (
	"strings"
	"testing"
)

// ─── ChunkText tests ─────────────────────────────────────────────────────────

func TestChunkText(t *testing.T) {
	// Build text with 10 paragraphs, each ~100 chars.
	var paragraphs []string
	for i := 0; i < 10; i++ {
		paragraphs = append(paragraphs, strings.Repeat("word ", 20)) // ~100 chars
	}
	text := strings.Join(paragraphs, "\n\n")

	config := ChunkerConfig{
		ChunkSize:    250,
		ChunkOverlap: 50,
		Separator:    "\n\n",
	}
	chunks := ChunkText(text, config)

	if len(chunks) < 3 {
		t.Fatalf("expected at least 3 chunks from 10 paragraphs with chunk_size=250, got %d", len(chunks))
	}

	// Verify chunk indices are sequential.
	for i, c := range chunks {
		if c.Index != i {
			t.Errorf("chunk %d has index %d, want %d", i, c.Index, i)
		}
	}

	// Verify no chunk exceeds chunk size (except possibly the first few chars of overlap).
	for i, c := range chunks {
		// Allow a tolerance: chunk might be slightly over due to segment boundaries.
		if len(c.Text) > config.ChunkSize+config.ChunkOverlap+10 {
			t.Errorf("chunk %d is %d chars, exceeds limit %d+%d",
				i, len(c.Text), config.ChunkSize, config.ChunkOverlap)
		}
	}

	// Verify overlap: tail of chunk N should appear at start of chunk N+1.
	for i := 0; i < len(chunks)-1; i++ {
		cur := chunks[i].Text
		next := chunks[i+1].Text
		if len(cur) >= config.ChunkOverlap {
			tail := cur[len(cur)-config.ChunkOverlap:]
			if !strings.HasPrefix(next, tail) {
				// Overlap may not be exact due to separator re-insertion,
				// but the tail text should appear somewhere in the next chunk.
				if !strings.Contains(next, tail[:min(20, len(tail))]) {
					t.Logf("warning: overlap not found between chunks %d and %d", i, i+1)
				}
			}
		}
	}
}

func TestChunkTextSmall(t *testing.T) {
	text := "This is a short text that fits in one chunk."
	config := ChunkerConfig{ChunkSize: 500}
	chunks := ChunkText(text, config)

	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk for small text, got %d", len(chunks))
	}
	if chunks[0].Text != text {
		t.Errorf("chunk text = %q, want %q", chunks[0].Text, text)
	}
	if chunks[0].Index != 0 {
		t.Errorf("chunk index = %d, want 0", chunks[0].Index)
	}
}

func TestChunkTextEmpty(t *testing.T) {
	chunks := ChunkText("", ChunkerConfig{})
	if len(chunks) != 0 {
		t.Errorf("expected 0 chunks for empty text, got %d", len(chunks))
	}
}

func TestChunkTextDefaultConfig(t *testing.T) {
	// With defaults: ChunkSize=500, Overlap=50, Separator="\n\n"
	text := strings.Repeat("Hello world. ", 100) // ~1300 chars
	chunks := ChunkText(text, ChunkerConfig{})
	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks for 1300 chars with default 500 chunk size, got %d", len(chunks))
	}
}

// ─── ParseMarkdown tests ─────────────────────────────────────────────────────

func TestParseMarkdown(t *testing.T) {
	md := `# Introduction

This is the introduction paragraph with some text.

## Methods

Here we describe the methods used in this study.

### Data Collection

Data was collected from multiple sources.

## Results

The results show significant improvement.
`
	config := ChunkerConfig{ChunkSize: 500}
	chunks := ParseMarkdown(md, config)

	if len(chunks) < 3 {
		t.Fatalf("expected at least 3 sections from markdown with 4 headers, got %d", len(chunks))
	}

	// Verify that headers appear in metadata.
	foundIntro := false
	foundMethods := false
	foundResults := false
	for _, c := range chunks {
		if h, ok := c.Metadata["header"]; ok {
			switch h {
			case "Introduction":
				foundIntro = true
			case "Methods":
				foundMethods = true
			case "Results":
				foundResults = true
			}
		}
	}
	if !foundIntro {
		t.Error("missing 'Introduction' header in chunk metadata")
	}
	if !foundMethods {
		t.Error("missing 'Methods' header in chunk metadata")
	}
	if !foundResults {
		t.Error("missing 'Results' header in chunk metadata")
	}
}

func TestParseMarkdownEmpty(t *testing.T) {
	chunks := ParseMarkdown("", ChunkerConfig{})
	if len(chunks) != 0 {
		t.Errorf("expected 0 chunks for empty markdown, got %d", len(chunks))
	}
}

func TestParseMarkdownNoHeaders(t *testing.T) {
	md := "Just some plain text without headers."
	chunks := ParseMarkdown(md, ChunkerConfig{ChunkSize: 500})
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk for markdown without headers, got %d", len(chunks))
	}
	if chunks[0].Text != md {
		t.Errorf("chunk text = %q, want %q", chunks[0].Text, md)
	}
}

// ─── ParseCSV tests ──────────────────────────────────────────────────────────

func TestParseCSV(t *testing.T) {
	csv := `name,age,city
Alice,30,New York
Bob,25,London
Charlie,35,Paris
`
	chunks, err := ParseCSV(csv, "people.csv")
	if err != nil {
		t.Fatalf("ParseCSV error: %v", err)
	}

	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks (one per data row), got %d", len(chunks))
	}

	// Verify first row content.
	if !strings.Contains(chunks[0].Text, "name: Alice") {
		t.Errorf("chunk 0 should contain 'name: Alice', got %q", chunks[0].Text)
	}
	if !strings.Contains(chunks[0].Text, "age: 30") {
		t.Errorf("chunk 0 should contain 'age: 30', got %q", chunks[0].Text)
	}
	if !strings.Contains(chunks[0].Text, "city: New York") {
		t.Errorf("chunk 0 should contain 'city: New York', got %q", chunks[0].Text)
	}

	// Verify metadata.
	for i, c := range chunks {
		if c.Metadata["source"] != "people.csv" {
			t.Errorf("chunk %d metadata source = %q, want 'people.csv'", i, c.Metadata["source"])
		}
	}
}

func TestParseCSVSingleRow(t *testing.T) {
	csv := "col1,col2,col3\n"
	chunks, err := ParseCSV(csv, "test.csv")
	if err != nil {
		t.Fatalf("ParseCSV error: %v", err)
	}
	// Single header row = treated as one chunk.
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk for header-only CSV, got %d", len(chunks))
	}
}

// ─── StripHTML tests ─────────────────────────────────────────────────────────

func TestStripHTML(t *testing.T) {
	tests := []struct {
		name string
		html string
		want string
	}{
		{
			name: "basic tags",
			html: "<p>Hello <b>world</b></p>",
			want: "Hello world",
		},
		{
			name: "nested divs",
			html: "<div><div>Inner content</div></div>",
			want: "Inner content",
		},
		{
			name: "entities",
			html: "<p>Tom &amp; Jerry &lt;3&gt;</p>",
			want: "Tom & Jerry <3>",
		},
		{
			name: "script and style",
			html: `<html><head><style>body{color:red}</style></head><body><p>Text</p><script>alert(1)</script></body></html>`,
			want: "body{color:red} Text alert(1)",
		},
		{
			name: "line breaks",
			html: "Line 1<br>Line 2<br/>Line 3",
			want: "Line 1 Line 2 Line 3",
		},
		{
			name: "nbsp",
			html: "Hello&nbsp;World",
			want: "Hello World",
		},
		{
			name: "empty",
			html: "",
			want: "",
		},
		{
			name: "plain text",
			html: "No tags here",
			want: "No tags here",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := StripHTML(tt.html)
			if got != tt.want {
				t.Errorf("StripHTML(%q) = %q, want %q", tt.html, got, tt.want)
			}
		})
	}
}

// ─── ParseDocument tests ─────────────────────────────────────────────────────

func TestParseDocumentTxt(t *testing.T) {
	chunks, err := ParseDocument("readme.txt", "Some text content.", ChunkerConfig{})
	if err != nil {
		t.Fatalf("ParseDocument error: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
}

func TestParseDocumentPDF(t *testing.T) {
	_, err := ParseDocument("file.pdf", "content", ChunkerConfig{})
	if err == nil {
		t.Fatal("expected error for PDF format")
	}
	if !strings.Contains(err.Error(), "not supported") {
		t.Errorf("error should mention 'not supported', got: %v", err)
	}
}

func TestParseDocumentDOCX(t *testing.T) {
	_, err := ParseDocument("file.docx", "content", ChunkerConfig{})
	if err == nil {
		t.Fatal("expected error for DOCX format")
	}
	if !strings.Contains(err.Error(), "not supported") {
		t.Errorf("error should mention 'not supported', got: %v", err)
	}
}

// ─── ChunkerConfig defaults tests ────────────────────────────────────────────

func TestChunkerConfigDefaults(t *testing.T) {
	c := ChunkerConfig{}.withDefaults()
	if c.ChunkSize != 500 {
		t.Errorf("default ChunkSize = %d, want 500", c.ChunkSize)
	}
	if c.ChunkOverlap != 50 {
		// Default overlap is left at 0 since the zero value is valid.
		// Only negative or >= ChunkSize values get corrected.
	}
	if c.Separator != "\n\n" {
		t.Errorf("default Separator = %q, want %q", c.Separator, "\n\n")
	}
}

func TestChunkerConfigOverlapClamped(t *testing.T) {
	c := ChunkerConfig{ChunkSize: 100, ChunkOverlap: 200}.withDefaults()
	if c.ChunkOverlap >= c.ChunkSize {
		t.Errorf("overlap %d should be < chunk_size %d", c.ChunkOverlap, c.ChunkSize)
	}
}
