/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "scheduler/request_output.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/threadpool.h"

namespace xllm_service {
namespace {

llm::RequestOutput cancellation() {
  llm::RequestOutput output;
  output.status = llm::Status(llm::StatusCode::CANCELLED, "instance failed");
  return output;
}

TEST(RequestOutputTest, ErrorTerminatesExactlyOnce) {
  Request request;
  int32_t callbacks = 0;
  request.output_callback = [&callbacks](const llm::RequestOutput&) {
    ++callbacks;
    return true;
  };
  EXPECT_EQ(deliver_request_output(request, cancellation()), true);
  EXPECT_FALSE(deliver_request_output(request, cancellation()).has_value());
  EXPECT_FALSE(
      deliver_request_output(request, llm::RequestOutput{}).has_value());
  EXPECT_EQ(callbacks, 1);
}

TEST(RequestOutputTest, SuccessfulTerminalOutputSuppressesLaterCancellation) {
  Request request;
  int32_t callbacks = 0;
  request.output_callback = [&callbacks](const llm::RequestOutput&) {
    ++callbacks;
    return true;
  };
  llm::RequestOutput output;
  output.finished = true;
  EXPECT_EQ(deliver_request_output(request, output), false);
  EXPECT_FALSE(deliver_request_output(request, cancellation()).has_value());
  EXPECT_EQ(callbacks, 1);
}

TEST(RequestOutputTest, CallbackFailureSuppressesRemainingOutputs) {
  Request request;
  int32_t callbacks = 0;
  request.output_callback = [&callbacks](const llm::RequestOutput&) {
    ++callbacks;
    return false;
  };
  EXPECT_EQ(deliver_request_output(request, llm::RequestOutput{}), true);
  EXPECT_FALSE(deliver_request_output(request, cancellation()).has_value());
  EXPECT_EQ(callbacks, 1);
}

TEST(RequestOutputTest, CancellationWaitsForQueuedGenerationAndCleansUpOnce) {
  auto request = std::make_shared<Request>();
  request->service_request_id = "request-with-owned-id";
  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::promise<void> released;
  auto released_future = released.get_future();
  std::vector<std::string> events;
  request->output_callback = [&](const llm::RequestOutput& output) {
    if (!output.status.has_value()) {
      events.emplace_back("generation-start");
      entered.set_value();
      released_future.wait();
      events.emplace_back("generation-end");
    } else {
      events.emplace_back("cancellation");
    }
    return true;
  };
  std::vector<std::pair<std::string, bool>> cleanups;
  ThreadPool pool;
  auto enqueue = [&](llm::RequestOutput output) {
    pool.schedule([request, output = std::move(output), &cleanups] {
      const std::optional<bool> finished =
          deliver_request_output(*request, output);
      if (finished.has_value()) {
        cleanups.emplace_back(request->service_request_id, *finished);
      }
    });
  };
  enqueue(llm::RequestOutput{});
  const auto started = entered_future.wait_for(std::chrono::seconds(5));
  if (started != std::future_status::ready) {
    released.set_value();
    FAIL() << "The output worker did not start";
  }
  enqueue(cancellation());
  enqueue(cancellation());
  enqueue(llm::RequestOutput{});
  released.set_value();
  std::promise<void> drained;
  auto drained_future = drained.get_future();
  pool.schedule([&drained] { drained.set_value(); });
  ASSERT_EQ(drained_future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  EXPECT_EQ(events,
            (std::vector<std::string>{
                "generation-start", "generation-end", "cancellation"}));
  EXPECT_EQ(cleanups,
            (std::vector<std::pair<std::string, bool>>{
                {"request-with-owned-id", true}}));
}

}  // namespace
}  // namespace xllm_service
