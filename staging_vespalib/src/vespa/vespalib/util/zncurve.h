// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <cstdint>
#include <vector>

namespace vespalib {

template<typename T>
class ZNPoint {
public:
    ZNPoint(const T * begin, uint32_t numDim);
    ZNPoint(ZNPoint && rhs) = default;
    ZNPoint & operator = (ZNPoint && rhs) = default;
    ~ZNPoint();
    bool operator < (const ZNPoint & rhs) const;
    double distance(const ZNPoint & rhs) const;
    uint32_t numDim() const { return _vector.size(); }
private:
    std::vector<T>       _vector;
    std::vector<uint8_t> _point;
};

extern template class ZNPoint<uint32_t>;
extern template class ZNPoint<float>;

} // namespace vespalib

