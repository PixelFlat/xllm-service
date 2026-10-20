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

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "common/concurrent_queue.h"
#include "tokenizer/tokenizer.h"

namespace xllm_service {

// Instances are initialized eagerly and borrowed exclusively for one operation.
// Exhaustion waits for an existing instance instead of cloning on a request.
class TokenizerPool final {
 public:
  TokenizerPool(std::unique_ptr<Tokenizer> tokenizer, size_t size);

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const;
  std::string decode(const Slice<int32_t>& ids, bool skip_special_tokens) const;

 private:
  class Lease;
  Lease acquire() const;

  mutable ConcurrentQueue<std::unique_ptr<Tokenizer>> available_;
};

}  // namespace xllm_service
