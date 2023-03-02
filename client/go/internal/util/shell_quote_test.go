// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package util

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestShellQuote(t *testing.T) {
	assert.Equal(t, "foo", ShellQuote("foo"))
	assert.Equal(t, "'foo bar'", ShellQuote("foo bar"))
	assert.Equal(t, "foo-bar", ShellQuote("foo-bar"))
	assert.Equal(t, "foo_bar", ShellQuote("foo_bar"))
	assert.Equal(t, "' foo'", ShellQuote(" foo"))
	assert.Equal(t, "'foo '", ShellQuote("foo "))
	assert.Equal(t, "' foo '", ShellQuote(" foo "))
	assert.Equal(t, `'a'\''b'`, ShellQuote("a'b"))
	assert.Equal(t, `''\'''`, ShellQuote("'"))
	assert.Equal(t, `'"'`, ShellQuote(`"`))
	assert.Equal(t, `'z?z?z?z'`, ShellQuote("z\u2318z\tz\rz"))
}

func TestShellQuoteArgs(t *testing.T) {
	assert.Equal(t, "foo 123 bar", ShellQuoteArgs("foo", "123", "bar"))
	assert.Equal(t, "'fo oo' '12 3' 'b ar'", ShellQuoteArgs("fo oo", "12 3", "b ar"))
}
