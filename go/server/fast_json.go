package server

import (
	"net/http"
	"strconv"
	"sync"
	"time"
)

// Pre-allocated buffer pool for JSON response writing.
// Avoids reflection-based json.Encoder and per-request allocation.
var jsonBufPool = sync.Pool{
	New: func() interface{} {
		b := make([]byte, 0, 4096)
		return &b
	},
}

// writeChatCompletionFast writes a ChatCompletionResponse without json.Encoder.
// Saves ~0.5ms per request vs reflection-based encoding.
func writeChatCompletionFast(w http.ResponseWriter, id string, model string, content string, promptToks, genToks int) {
	bp := jsonBufPool.Get().(*[]byte)
	b := (*bp)[:0]

	b = append(b, `{"id":"`...)
	b = append(b, id...)
	b = append(b, `","object":"chat.completion","created":`...)
	b = strconv.AppendInt(b, time.Now().Unix(), 10)
	b = append(b, `,"model":"`...)
	b = append(b, model...)
	b = append(b, `","choices":[{"index":0,"message":{"role":"assistant","content":`...)
	b = appendJSONString(b, content)
	b = append(b, `},"finish_reason":"stop"}],"usage":{"prompt_tokens":`...)
	b = strconv.AppendInt(b, int64(promptToks), 10)
	b = append(b, `,"completion_tokens":`...)
	b = strconv.AppendInt(b, int64(genToks), 10)
	b = append(b, `,"total_tokens":`...)
	b = strconv.AppendInt(b, int64(promptToks+genToks), 10)
	b = append(b, `}}`...)
	b = append(b, '\n')

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(b) //nolint:errcheck

	*bp = b
	jsonBufPool.Put(bp)
}

// buildChatCompletionJSON returns a pre-serialized JSON response as a byte slice.
// Used by the response cache so the same bytes can be served on cache hits.
func buildChatCompletionJSON(id string, model string, content string, promptToks, genToks int) []byte {
	bp := jsonBufPool.Get().(*[]byte)
	b := (*bp)[:0]

	b = append(b, `{"id":"`...)
	b = append(b, id...)
	b = append(b, `","object":"chat.completion","created":`...)
	b = strconv.AppendInt(b, time.Now().Unix(), 10)
	b = append(b, `,"model":"`...)
	b = append(b, model...)
	b = append(b, `","choices":[{"index":0,"message":{"role":"assistant","content":`...)
	b = appendJSONString(b, content)
	b = append(b, `},"finish_reason":"stop"}],"usage":{"prompt_tokens":`...)
	b = strconv.AppendInt(b, int64(promptToks), 10)
	b = append(b, `,"completion_tokens":`...)
	b = strconv.AppendInt(b, int64(genToks), 10)
	b = append(b, `,"total_tokens":`...)
	b = strconv.AppendInt(b, int64(promptToks+genToks), 10)
	b = append(b, `}}`...)
	b = append(b, '\n')

	// Copy to a new slice so the pooled buffer can be returned.
	out := make([]byte, len(b))
	copy(out, b)

	*bp = b
	jsonBufPool.Put(bp)
	return out
}

// appendJSONString appends a JSON-escaped string to b.
func appendJSONString(b []byte, s string) []byte {
	b = append(b, '"')
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '"':
			b = append(b, '\\', '"')
		case '\\':
			b = append(b, '\\', '\\')
		case '\n':
			b = append(b, '\\', 'n')
		case '\r':
			b = append(b, '\\', 'r')
		case '\t':
			b = append(b, '\\', 't')
		default:
			if c < 0x20 {
				b = append(b, '\\', 'u', '0', '0',
					"0123456789abcdef"[c>>4],
					"0123456789abcdef"[c&0xf])
			} else {
				b = append(b, c)
			}
		}
	}
	b = append(b, '"')
	return b
}

// fastID generates a request ID without fmt.Sprintf allocation.
func fastID(prefix string) string {
	b := make([]byte, 0, len(prefix)+20)
	b = append(b, prefix...)
	b = append(b, '-')
	b = strconv.AppendInt(b, time.Now().UnixNano(), 10)
	return string(b)
}
