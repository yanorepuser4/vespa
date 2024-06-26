// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#pragma once

#include <vespamalloc/malloc/memblockboundscheck.h>

namespace vespamalloc {

typedef MemBlockBoundsCheckBaseT<20, 0> MemBlockBoundsCheckBase;

class MemBlockBoundsCheck : public MemBlockBoundsCheckBase
{
public:
    MemBlockBoundsCheck() : MemBlockBoundsCheckBase() { }
    MemBlockBoundsCheck(void * p) : MemBlockBoundsCheckBase(p) { }
    MemBlockBoundsCheck(void * p, size_t sz) : MemBlockBoundsCheckBase(p, sz) { }
    MemBlockBoundsCheck(void * p, size_t sz, bool dummy) : MemBlockBoundsCheckBase(p, sz, dummy) { }
    bool validAlloc() const { return validAlloc1(); }
    bool validFree()  const { return validFree1(); }
};

}

