# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
vespa_add_executable(searchcore_document_db_reference_resolver_test_app TEST
    SOURCES
    document_db_reference_resolver_test.cpp
    DEPENDS
    searchcore_reference
    searchcore_attribute
    searchcore_test
    searchlib_test
)
vespa_add_test(NAME searchcore_document_db_reference_resolver_test_app COMMAND searchcore_document_db_reference_resolver_test_app)
