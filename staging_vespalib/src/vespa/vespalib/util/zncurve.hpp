// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include "zncurve.h"
#include <cstring>
#include <cmath>
#include <cassert>

namespace vespalib {

namespace {

template<typename T>
double squareDistance(T a, T b) {
    T diff = a - b;
    return diff * diff;
}

template<uint32_t>
double squareDistance(uint32_t a, uint32_t b) {
    int64_t diff = a - b;
    return diff * diff;
}

}

template<typename T>
ZNPoint<T>::ZNPoint(const T * begin, uint32_t numDim)
    : _vector(numDim),
      _point(numDim*sizeof(T))
{
    for (uint32_t i(0); i < numDim; i++) {
        _vector[i] = begin[i];
    }
}

template<typename T>
double
ZNPoint<T>::distance(const ZNPoint & rhs) const
{
    assert(numDim() == rhs.numDim());
    double sum(0);
    for (unsigned i(0); i < numDim(); i++) {
        sum += squareDistance(_vector[i], rhs._vector[i]);
    }
    return sqrt(sum);
}

template<typename T>
bool
ZNPoint<T>::operator < (const ZNPoint & rhs) const
{
    assert(numDim() == rhs.numDim());
    return memcmp(&_point[0], &rhs._point[0], _point.size()) < 0;
}

} // namespace vespalib

