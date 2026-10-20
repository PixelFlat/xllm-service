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

#include <glog/logging.h>

#include <utility>

namespace xllm_service {

class TokenizerPool::Lease final {
 public:
  Lease(const TokenizerPool& pool, std::unique_ptr<Tokenizer> tokenizer)
      : pool_(pool), tokenizer_(std::move(tokenizer)) {}
  ~Lease() { pool_.available_.push(std::move(tokenizer_)); }

  Lease(const Lease&) = delete;
  Lease& operator=(const Lease&) = delete;
  Lease(Lease&&) = delete;
  Lease& operator=(Lease&&) = delete;

  const Tokenizer* operator->() const { return tokenizer_.get(); }

 private:
  const TokenizerPool& pool_;
  std::unique_ptr<Tokenizer> tokenizer_;
};

TokenizerPool::TokenizerPool(std::unique_ptr<Tokenizer> tokenizer,
                             size_t size) {
  CHECK(tokenizer != nullptr);
  CHECK_GT(size, 0);
  for (size_t index = 1; index < size; ++index) {
    auto instance = tokenizer->clone();
    CHECK(instance != nullptr);
    instance->warmup();
    available_.push(std::move(instance));
  }
  tokenizer->warmup();
  available_.push(std::move(tokenizer));
}

TokenizerPool::Lease TokenizerPool::acquire() const {
  return Lease(*this, available_.pop());
}

bool TokenizerPool::encode(const std::string_view& text,
                           std::vector<int32_t>* ids,
                           bool add_special_tokens) const {
  auto tokenizer = acquire();
  return tokenizer->encode(text, ids, add_special_tokens);
}

std::string TokenizerPool::decode(const Slice<int32_t>& ids,
                                  bool skip_special_tokens) const {
  auto tokenizer = acquire();
  return tokenizer->decode(ids, skip_special_tokens);
}

}  // namespace xllm_service
