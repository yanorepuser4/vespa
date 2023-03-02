// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
// load default environment variables (from $VESPA_HOME/conf/vespa/default-env.txt)
// Author: arnej

package util

import (
	"bytes"
)

func needQuoting(s string) bool {
	for _, ch := range s {
		switch {
		case (ch >= 'A' && ch <= 'Z'):
		case (ch >= 'a' && ch <= 'z'):
		case (ch >= '0' && ch <= '9'):
		case ch == '-':
		case ch == '_':
		case ch == '.':
		case ch == ':':
		case ch == '/':
			// all above: nop
		default:
			return true
		}
	}
	return false
}

func quote(s string, t *bytes.Buffer) {
	const singleQuote = '\''
	if !needQuoting(s) {
		t.WriteString(s)
		return
	}
	t.WriteByte(singleQuote)
	for _, ch := range s {
		switch {
		case ch == '\'' || ch == '\\':
			t.WriteByte(singleQuote)
			t.WriteByte('\\')
			t.WriteByte(byte(ch))
			t.WriteByte(singleQuote)
		case ch < 32 || ch > 127:
			t.WriteByte('?')
		default:
			t.WriteByte(byte(ch))
		}
	}
	t.WriteByte(singleQuote)
}

func ShellQuoteArgs(args ...string) string {
	var buf bytes.Buffer
	for idx, s := range args {
		if idx > 0 {
			buf.WriteByte(' ')
		}
		quote(s, &buf)
	}
	return buf.String()
}

func ShellQuote(s string) string {
	var buf bytes.Buffer
	quote(s, &buf)
	return buf.String()
}
