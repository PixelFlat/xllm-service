include(cc_test)

cc_test(
  NAME
    request_metrics_test
  SRCS
    request_metrics_test.cpp
  DEPS
    :request
    GTest::gtest_main
)

cc_test(
  NAME
    response_handler_test
  SRCS
    response_handler_test.cpp
  DEPS
    :response_handler
    GTest::gtest_main
)
