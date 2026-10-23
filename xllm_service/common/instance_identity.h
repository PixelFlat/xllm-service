/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace xllm_service {

struct InstanceIdentity {
  std::string name;
  std::string incarnation_id;
  uint64_t register_ts_ms = 0;
};

inline bool same_instance_identity(const InstanceIdentity& selected,
                                   std::string_view name,
                                   std::string_view incarnation_id,
                                   uint64_t register_ts_ms) {
  return selected.name == name && selected.incarnation_id == incarnation_id &&
         (!incarnation_id.empty() || selected.register_ts_ms == register_ts_ms);
}

inline bool same_instance_identity(const InstanceIdentity& left,
                                   const InstanceIdentity& right) {
  return same_instance_identity(
      left, right.name, right.incarnation_id, right.register_ts_ms);
}

}  // namespace xllm_service
