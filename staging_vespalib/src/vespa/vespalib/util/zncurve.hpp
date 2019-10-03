// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include "zncurve.h"
#include "sort.h"
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

uint32_t byteIndex(uint32_t index) { return index >> 3; }
uint8_t mask(uint32_t index) { return 1 << (7 - (index % 8)); }

}

template<typename T>
ZNPoint<T>::ZNPoint(const T * begin, uint32_t numDim)
    : _vector(numDim),
      _point(numDim*sizeof(T))
{
    using Converter = convertForSort<T, true>;
    using ConvertedT = typename Converter::UIntType;
    Converter converter;
    assert(sizeof(T) == sizeof(ConvertedT));
    std::vector<ConvertedT> converted(numDim);
    for (uint32_t i(0); i < numDim; i++) {
        _vector[i] = begin[i];
        converted[i] = converter.convert(begin[i]);
    }

    uint32_t bitPos = _point.size()*8;
    for (uint32_t bitNum(0); bitNum < sizeof(ConvertedT)*8; bitNum++) {
        for (uint32_t dim(converted.size()); dim; dim--) {
           ConvertedT tmp = converted[dim - 1];
           bitPos--;
           if (tmp & (1 << bitNum)) {
               _point[byteIndex(bitPos)] |= mask(bitPos);
           }
        }
    }
    assert(bitPos == 0);
}

template<typename T>
ZNPoint<T>::~ZNPoint() = default;

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

