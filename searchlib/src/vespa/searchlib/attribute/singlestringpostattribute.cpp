// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "singlestringpostattribute.hpp"
#include "postinglistsearchcontext.hpp"


namespace search {

template class SingleValueStringPostingAttributeT<EnumAttribute<StringAttribute>>;

} // namespace search

