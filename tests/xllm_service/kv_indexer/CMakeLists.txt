include(cc_test)

cc_test(
  NAME
    kv_directory_test
  SRCS
    kv_directory_test.cpp
  DEPS
    :kv_directory
    :loadbalance_policy
    :xllm_http_service
    GTest::gtest_main
)
