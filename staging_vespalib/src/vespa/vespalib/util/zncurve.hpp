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

template<>
double squareDistance(uint32_t a32, uint32_t b32) {
    int64_t a = a32, b = b32;
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
        uint32_t dim(converted.size());
        for (; dim >= 8; dim -= 8) {
            bitPos -= 8;
            _point[byteIndex(bitPos)] =
                 ((converted[dim - 1] >> bitNum) & 0x01l) |
                (((converted[dim - 2] >> bitNum) & 0x01l) << 1)|
                (((converted[dim - 3] >> bitNum) & 0x01l) << 2)|
                (((converted[dim - 4] >> bitNum) & 0x01l) << 3)|
                (((converted[dim - 5] >> bitNum) & 0x01l) << 4)|
                (((converted[dim - 6] >> bitNum) & 0x01l) << 5)|
                (((converted[dim - 7] >> bitNum) & 0x01l) << 6)|
                (((converted[dim - 8] >> bitNum) & 0x01l) << 7);
        }
        for (; dim; dim--) {
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
ZNPoint<T>::ZNPoint(const ZNPoint &) = default;
template<typename T>
ZNPoint<T> & ZNPoint<T>::operator = (const ZNPoint &) = default;

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
void
ZNPoint<T>::ceil(uint32_t numLSB)
{
    uint32_t numBits2Set = numLSB * numDim();
    uint32_t numBytes2Set = numBits2Set / 8;
    uint32_t startByte = _point.size() - numBytes2Set;
    for (uint32_t byteIdx(startByte); byteIdx < _point.size(); byteIdx++) {
        _point[byteIdx] = 0xff;
    }
    for (uint32_t i(numBytes2Set * 8); i < numBits2Set; i++) {
        uint32_t bitPos = (_point.size() * 8) - i;
        _point[byteIndex(bitPos)] |= mask(bitPos);
    }
}

template<typename T>
void
ZNPoint<T>::floor(uint32_t numLSB)
{
    uint32_t numBits2Set = numLSB * numDim();
    uint32_t numBytes2Set = numBits2Set / 8;
    uint32_t startByte = _point.size() - numBytes2Set;
    for (uint32_t byteIdx(startByte); byteIdx < _point.size(); byteIdx++) {
        _point[byteIdx] = 0x00;
    }
    for (uint32_t i(numBytes2Set * 8); i < numBits2Set; i++) {
        uint32_t bitPos = (_point.size() * 8) - i;
        _point[byteIndex(bitPos)] &= ~mask(bitPos);
    }
}

template<typename T>
bool
ZNPoint<T>::operator < (const ZNPoint & rhs) const
{
    assert(numDim() == rhs.numDim());
    return memcmp(&_point[0], &rhs._point[0], _point.size()) < 0;
}

} // namespace vespalib

