// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
// Unit tests for templatetermvisitor.

#include <vespa/searchlib/query/tree/intermediatenodes.h>
#include <vespa/searchlib/query/tree/templatetermvisitor.h>
#include <vespa/searchlib/query/tree/simplequery.h>
#include <vespa/searchlib/query/tree/termnodes.h>
#include <vespa/vespalib/gtest/gtest.h>

using namespace search::query;

namespace {

class MyVisitor;

class MyVisitor : public TemplateTermVisitor<MyVisitor, SimpleQueryNodeTypes>
{
public:
    template <typename T>
    bool &isVisited() {
        static bool b;
        return b;
    }

    template <class TermType>
    void visitTerm(TermType &) { isVisited<TermType>() = true; }
};

template <class T>
bool checkVisit(T *q) {
    Node::UP query(q);
    MyVisitor visitor;
    visitor.isVisited<T>() = false;
    query->accept(visitor);
    return visitor.isVisited<T>();
}

template <class T>
bool checkVisit() {
    return checkVisit(new T(typename T::Type(), "field", 0, Weight(0)));
}

TEST(TemplateTermVisitorTest, require_that_all_terms_can_be_visited)
{
    EXPECT_TRUE(checkVisit<SimpleNumberTerm>());
    EXPECT_TRUE(checkVisit<SimpleLocationTerm>());
    EXPECT_TRUE(checkVisit<SimplePrefixTerm>());
    EXPECT_TRUE(checkVisit<SimpleRangeTerm>());
    EXPECT_TRUE(checkVisit<SimpleStringTerm>());
    EXPECT_TRUE(checkVisit<SimpleSubstringTerm>());
    EXPECT_TRUE(checkVisit<SimpleSuffixTerm>());
    EXPECT_TRUE(checkVisit<SimplePredicateQuery>());
    EXPECT_TRUE(checkVisit<SimpleRegExpTerm>());
    EXPECT_TRUE(checkVisit(new SimplePhrase("field", 0, Weight(0))));
    EXPECT_TRUE(checkVisit(new SimpleSameElement("foo", 0, Weight(0))));
    EXPECT_TRUE(!checkVisit(new SimpleAnd));
    EXPECT_TRUE(!checkVisit(new SimpleAndNot));
    EXPECT_TRUE(!checkVisit(new SimpleEquiv(17, Weight(100))));
    EXPECT_TRUE(!checkVisit(new SimpleNear(2)));
    EXPECT_TRUE(!checkVisit(new SimpleONear(2)));
    EXPECT_TRUE(!checkVisit(new SimpleOr));
    EXPECT_TRUE(!checkVisit(new SimpleRank));
}

}  // namespace

GTEST_MAIN_RUN_ALL_TESTS()
