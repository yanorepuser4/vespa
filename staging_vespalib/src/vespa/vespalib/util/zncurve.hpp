// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include "zncurve.h"
#include <cstring>
#include <cmath>

namespace vespalib {

template<int numDim, class T>
ZNPoint<numDim, T>::ZNPoint(const T * begin)
    : _vector(),
      _zCurve()
{
    for (uint32_t i(0); i < numDim; i++) {
        _vector[i] = begin[i];
    }
}

template<int numDim, class T>
uint64_t
ZNPoint<numDim, T>::distance(const ZNPoint & rhs) const
{
    uint64_t sum(0);
    for (unsigned i(0); i < numDim; i++) {
        int64_t diff = _vector[i] - rhs._vector[i];
        sum += diff*diff;
    }
    return sqrt(sum);
}

template<int numDim, class T>
bool
ZNPoint<numDim, T>::operator < (const ZNPoint & rhs) const
{
    return memcmp(_zCurve, rhs._zCurve, sizeof(_zCurve)) < 0;
}

} // namespace vespalib

