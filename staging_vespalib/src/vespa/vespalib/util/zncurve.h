// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <cstdint>

namespace vespalib {

template<int numDim, class T>
class ZNPoint {
public:
    ZNPoint(const T * begin);
    uint64_t distance(const ZNPoint & rhs) const;
    bool operator < (const ZNPoint & rhs) const;
private:
    T       _vector[numDim];
    uint8_t _zCurve[sizeof(T)*numDim];
};

} // namespace vespalib

