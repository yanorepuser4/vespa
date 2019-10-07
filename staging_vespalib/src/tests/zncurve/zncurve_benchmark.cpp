// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/util/zncurve.h>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <fstream>

using namespace vespalib;

using ZPoint = ZNPoint<uint32_t>;

ZPoint
create_point(uint32_t numDims) {
    std::vector<uint32_t> vector(numDims);
    for (uint32_t & value : vector) {
        value = rand()%3000000;
    }
    return ZPoint(&vector[0], numDims);
}

std::vector<ZPoint>
create_points(uint32_t numPoints, uint32_t numDims) {
    std::vector<ZPoint> points;
    points.reserve(numPoints);
    std::vector<uint32_t> vector(numDims);
    for (uint32_t i(0); i < numPoints; i++) {
        for (uint32_t & value : vector) {
            value = rand();
        }
        points.emplace_back(&vector[0], numDims);
    }
    return points;
}

std::vector<ZPoint>
read_points(std::string fileName, uint32_t numDims) {
    std::vector<ZPoint> points;
    std::vector<uint32_t> vector(numDims);
    std::ifstream is(fileName);
    for (uint32_t line(0); is && ! is.eof(); line++) {
        char c, s, e, sep;
        int32_t key;
        int32_t signed_value;
        is >> c;
        //printf("Line: %5d c=%c\n", line, c);
        if ( c != '{' || is.eof()) { break; }
        
        for (uint32_t & value : vector) {
            is >> s >> key >> e >> c >> signed_value >> sep;
            signed_value += 1500000;

            if (signed_value < 0) {
                printf("Line: %5d key=%d value=%d\n", line, key, signed_value);
            }
            value = signed_value;
            assert(s == '"');
            assert(e == '"');
            assert(c == ':');
            assert((sep == ',') || (sep == '}'));
        }
        points.emplace_back(&vector[0], numDims);
    }
    return points;
}

void
analyzeByBits(const std::vector<ZPoint> & points, uint32_t analyzeCount) {
    //ZPoint center = create_point(points[0].numDim());
    ZPoint center = points[rand()%points.size()];
    auto lower = std::lower_bound(points.begin(), points.end(), center);
    printf("center is at pos %ld: ", lower - points.begin());
    for (uint32_t numLSB(1); numLSB <= 32; numLSB++) {
        ZPoint ceil = center;
        ceil.ceil(numLSB);
        ZPoint floor = center;
        floor.floor(numLSB);
        auto ceilIt = std::lower_bound(points.begin(), points.end(), ceil);
        auto floorIt = std::lower_bound(points.begin(), points.end(), floor);
        printf("%2d: (%5ld <= %5ld <= %5ld) = %5ld\n", numLSB, floorIt - points.begin(), lower - points.begin(), ceilIt - points.begin(), ceilIt - floorIt);
    }
    for (auto cur = lower; cur < lower+analyzeCount && cur < points.end(); cur++) {
        assert(! (*cur < center));
        printf("%1.4g ", center.distance(*cur));
    }
    printf("\n");
}

void
analyzeByBits(const std::vector<ZPoint> & points, uint32_t numLookups, uint32_t analyzeCount) {
    for (uint32_t i(0); i < numLookups; i++) {
        analyzeByBits(points, analyzeCount);
    }
}

void
verifyOrder(const std::vector<ZPoint> & points) {
    for (uint32_t i(1); i < points.size(); i++) {
        assert( ! (points[i] < points[i-1]) );
    }
}


int main (int argc, const char * argv[]) {
    assert(argc == 5);
    uint32_t numPoints = atoi(argv[1]);
    std::string fileName = argv[1];
    uint32_t numDims = atoi(argv[2]);
    uint32_t numLookups = atoi(argv[3]);
    uint32_t analyzeCount = atoi(argv[4]);

    std::vector<ZPoint> points = numPoints
                               ? create_points(numPoints, numDims)
                               : read_points(fileName, numDims);
    std::sort(points.begin(), points.end());
    verifyOrder(points);
    analyzeByBits(points, numLookups, analyzeCount);
    return 0;
}
