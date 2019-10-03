// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/testkit/testapp.h>
#include <vespa/vespalib/util/zncurve.h>

using namespace vespalib;

template <typename T>
void verifyDistance(double expected, std::vector<T> inputA, std::vector<T> inputB) {
    ZNPoint a(&inputA[0], inputA.size());
    ZNPoint b(&inputB[0], inputB.size());
    ASSERT_EQUAL(0.0, a.distance(a));
    ASSERT_EQUAL(0.0, b.distance(b));
    ASSERT_EQUAL(expected, a.distance(b));
    ASSERT_EQUAL(expected, b.distance(a));
}

template <typename T>
void verifyOrder(std::vector<T> first_in, std::vector<T> second_in, std::vector<T> third_in) {
    ZNPoint first(&first_in[0], first_in.size());
    ZNPoint second(&second_in[0], second_in.size());
    ZNPoint third(&third_in[0], third_in.size());
    ASSERT_TRUE(first < second);
    ASSERT_TRUE(second < third);
    ASSERT_TRUE(first < third);
    ASSERT_FALSE(second < first);
    ASSERT_FALSE(third < second);
    ASSERT_FALSE(third < first);
}

TEST("verify distance") {
   verifyDistance<uint32_t>(5, {3,7}, {7, 10});
   verifyDistance<float>(5, {3,7}, {7, 10});
}

TEST("verify order") {
   verifyOrder<uint32_t>({3}, {7}, {8});
   verifyOrder<uint32_t>({3,7}, {7, 10}, {8, 10});
   verifyOrder<uint32_t>({3, 2}, {3, 3}, {4, 1});
   verifyOrder<uint32_t>({3, 2, 3}, {3, 3, 2}, {3, 2, 7});

   verifyOrder<float>({3}, {7}, {8});
   verifyOrder<float>({3,7}, {7, 10}, {8, 10});
//TODO: Investigate below failure.
//   verifyOrder<float>({3, 2}, {3, 3}, {4, 1});
}

TEST_MAIN() { TEST_RUN_ALL(); }
