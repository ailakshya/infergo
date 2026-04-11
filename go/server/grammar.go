package server

import "context"

// grammarCtxKey is the context key for passing a GBNF grammar string
// from the HTTP handler through to the scheduler's generate loop.
type grammarCtxKey struct{}

// WithGrammar attaches a GBNF grammar string to a context.
func WithGrammar(ctx context.Context, grammar string) context.Context {
	return context.WithValue(ctx, grammarCtxKey{}, grammar)
}

// GrammarFromContext extracts the GBNF grammar from context, if present.
func GrammarFromContext(ctx context.Context) (string, bool) {
	g, ok := ctx.Value(grammarCtxKey{}).(string)
	return g, ok && g != ""
}

// JSONGrammar is a GBNF grammar that accepts any valid JSON value.
// Used for response_format: {"type": "json_object"}.
const JSONGrammar = `root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^\\"\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? (([eE] [-+]? [0-9]+))? ws

ws ::= ([ \t\n] ws)?
`
