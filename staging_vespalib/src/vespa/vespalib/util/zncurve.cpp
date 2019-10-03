// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "zncurve.hpp"

namespace vespalib {

template class ZNPoint<2, uint32_t>;
template class ZNPoint<4, uint32_t>;
template class ZNPoint<8, uint32_t>;

} // namespace vespalib

