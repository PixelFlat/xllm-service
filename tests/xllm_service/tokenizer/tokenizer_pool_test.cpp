/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tokenizer/tokenizer_pool.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

namespace xllm_service {
namespace {

struct State {
  std::atomic<int32_t> instances{0};
  std::atomic<int32_t> overlapping_calls{0};
  std::atomic<int32_t> calls_before_warmup{0};
  std::array<std::atomic<int32_t>, 4> warmup_flags{};
};

class RecordingTokenizer final : public Tokenizer {
 public:
  explicit RecordingTokenizer(std::shared_ptr<State> state, int32_t value = 7)
      : state_(std::move(state)),
        index_(state_->instances.fetch_add(1)),
        value_(value) {}

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const override {
    if (text == "warmup") {
      state_->warmup_flags[index_].fetch_or(1);
    } else {
      if (state_->warmup_flags[index_].load() != 7) {
        state_->calls_before_warmup.fetch_add(1);
      }
      record_call();
    }
    *ids = {value_, add_special_tokens ? 1 : 0};
    return text != "reject";
  }

  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override {
    state_->warmup_flags[index_].fetch_or(skip_special_tokens ? 4 : 2);
    record_call();
    return std::to_string(ids[0]);
  }

  std::optional<int32_t> token_to_id(const std::string_view&) const override {
    return std::nullopt;
  }
  std::string id_to_token(int32_t) const override { return ""; }
  size_t vocab_size() const override { return 10; }
  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<RecordingTokenizer>(state_, value_);
  }

 private:
  void record_call() const {
    if (active_.fetch_add(1) != 0) {
      state_->overlapping_calls.fetch_add(1);
    }
    std::this_thread::yield();
    active_.fetch_sub(1);
  }

  std::shared_ptr<State> state_;
  int32_t index_;
  int32_t value_;
  mutable std::atomic<int32_t> active_{0};
};

TEST(TokenizerPoolTest, InitializesEncodeAndDecodeBeforeServing) {
  auto state = std::make_shared<State>();
  TokenizerPool pool(std::make_unique<RecordingTokenizer>(state), /*size=*/3);
  EXPECT_EQ(state->instances, 3);
  for (int32_t index = 0; index < 3; ++index) {
    EXPECT_EQ(state->warmup_flags[index], 7);
  }
  std::vector<int32_t> ids;
  EXPECT_TRUE(pool.encode("request", &ids, /*add_special_tokens=*/false));
  EXPECT_EQ(ids, (std::vector<int32_t>{7, 0}));
  EXPECT_EQ(pool.decode(ids, /*skip_special_tokens=*/true), "7");
}

TEST(TokenizerPoolTest, ReusesExclusiveInstancesAcrossMoreThreadsThanCapacity) {
  auto state = std::make_shared<State>();
  TokenizerPool pool(std::make_unique<RecordingTokenizer>(state), /*size=*/2);
  std::vector<std::thread> workers;
  workers.reserve(8);
  for (int32_t index = 0; index < 8; ++index) {
    workers.emplace_back([&pool]() {
      for (int32_t request = 0; request < 50; ++request) {
        std::vector<int32_t> ids;
        EXPECT_TRUE(pool.encode("request", &ids));
        EXPECT_EQ(pool.decode(ids, /*skip_special_tokens=*/false), "7");
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  EXPECT_EQ(state->instances, 2);
  EXPECT_EQ(state->overlapping_calls, 0);
  EXPECT_EQ(state->calls_before_warmup, 0);
}

TEST(TokenizerPoolTest, ReleasesInstanceWhenEncodeReturnsFalse) {
  auto state = std::make_shared<State>();
  TokenizerPool pool(std::make_unique<RecordingTokenizer>(state), /*size=*/1);
  std::vector<int32_t> ids;
  EXPECT_FALSE(pool.encode("reject", &ids));
  EXPECT_TRUE(pool.encode("request", &ids));
  EXPECT_EQ(pool.decode(ids, /*skip_special_tokens=*/true), "7");
  EXPECT_EQ(state->instances, 1);
}

TEST(TokenizerPoolTest, KeepsDifferentModelsIsolatedOnTheSameThread) {
  auto first_state = std::make_shared<State>();
  auto second_state = std::make_shared<State>();
  TokenizerPool first(std::make_unique<RecordingTokenizer>(first_state, 3),
                      /*size=*/1);
  TokenizerPool second(std::make_unique<RecordingTokenizer>(second_state, 9),
                       /*size=*/1);
  std::vector<int32_t> ids;
  EXPECT_TRUE(first.encode("request", &ids));
  EXPECT_EQ(ids.front(), 3);
  EXPECT_TRUE(second.encode("request", &ids));
  EXPECT_EQ(ids.front(), 9);
}

}  // namespace
}  // namespace xllm_service
