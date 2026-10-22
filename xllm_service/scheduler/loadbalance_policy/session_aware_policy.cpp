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

#include "scheduler/loadbalance_policy/session_aware_policy.h"

#include <absl/strings/strip.h>
#include <glog/logging.h>

#include <algorithm>
#include <list>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

namespace xllm_service {
namespace {

struct Binding {
  RoutingCandidate prefill;
  std::optional<RoutingCandidate> decode;
  uint64_t version = 0;
  uint64_t active = 0;
  bool committed = false;
  SessionAwarePolicy::TimePoint idle_since;
  std::optional<std::list<std::string>::iterator> idle_position;
};

const RoutingCandidate* find_candidate(
    const std::vector<RoutingCandidate>& candidates,
    const std::string& name) {
  const auto found = std::find_if(
      candidates.begin(), candidates.end(), [&name](const auto& candidate) {
        return candidate.name == name;
      });
  return found == candidates.end() ? nullptr : &*found;
}

bool same_instance(const RoutingCandidate& left,
                   const RoutingCandidate& right) {
  return left.name == right.name &&
         left.incarnation_id == right.incarnation_id &&
         (!left.incarnation_id.empty() ||
          left.register_ts_ms == right.register_ts_ms);
}

const RoutingCandidate* bound_candidate(
    const std::vector<RoutingCandidate>& candidates,
    const RoutingCandidate& target) {
  const auto* found = find_candidate(candidates, target.name);
  return found != nullptr && same_instance(*found, target) ? found : nullptr;
}

bool set_route(const std::shared_ptr<Request>& request,
               InstanceMgr& manager,
               const RoutingCandidate* prefill,
               const RoutingCandidate* decode) {
  if (prefill == nullptr ||
      (decode == nullptr && prefill->type != InstanceType::DEFAULT)) {
    return false;
  }
  request->routing.prefill_name = prefill->name;
  request->routing.decode_name = decode == nullptr ? "" : decode->name;
  request->prefill_incarnation_id = prefill->incarnation_id;
  request->decode_incarnation_id =
      decode == nullptr ? "" : decode->incarnation_id;
  request->prefill_register_ts_ms = prefill->register_ts_ms;
  request->decode_register_ts_ms =
      decode == nullptr ? std::nullopt
                        : std::optional<uint64_t>(decode->register_ts_ms);
  return manager.bind_request_instance_incarnations(request);
}

}  // namespace

class SessionAffinityState final {
 public:
  SessionAffinityState(int32_t ttl_s,
                       int32_t max_entries,
                       SessionAwarePolicy::Clock clock)
      : ttl_(std::chrono::seconds(ttl_s)),
        max_entries_(static_cast<size_t>(max_entries)),
        clock_(std::move(clock)) {
    CHECK_GT(ttl_s, 0);
    CHECK_GT(max_entries, 0);
    CHECK(clock_);
  }

  void erase(const std::string& key) {
    const auto found = bindings_.find(key);
    if (found == bindings_.end()) {
      return;
    }
    if (found->second.idle_position.has_value()) {
      idle_.erase(*found->second.idle_position);
    }
    bindings_.erase(found);
  }

  void expire(SessionAwarePolicy::TimePoint now) {
    while (!idle_.empty() &&
           now - bindings_.at(idle_.front()).idle_since >= ttl_) {
      erase(idle_.front());
    }
  }

  std::mutex mutex_;
  const std::chrono::seconds ttl_;
  const size_t max_entries_;
  SessionAwarePolicy::Clock clock_;
  std::unordered_map<std::string, Binding> bindings_;
  // Only idle, committed bindings appear here; active requests cannot be
  // evicted.
  std::list<std::string> idle_;
  uint64_t next_version_ = 0;
};

SessionLease::SessionLease(std::shared_ptr<SessionAffinityState> state,
                           std::string key,
                           uint64_t version)
    : state_(std::move(state)), key_(std::move(key)), version_(version) {}

SessionLease::~SessionLease() { finish(/*failed=*/false); }

void SessionLease::commit() {
  if (committed_.load(std::memory_order_relaxed) ||
      finished_.load(std::memory_order_relaxed)) {
    return;
  }
  std::lock_guard<std::mutex> lock(state_->mutex_);
  const auto found = state_->bindings_.find(key_);
  if (!finished_.load(std::memory_order_relaxed) &&
      found != state_->bindings_.end() && found->second.version == version_) {
    found->second.committed = true;
    committed_.store(true, std::memory_order_relaxed);
  }
}

void SessionLease::finish(bool failed) {
  std::lock_guard<std::mutex> lock(state_->mutex_);
  if (finished_.exchange(true, std::memory_order_relaxed)) {
    return;
  }
  const auto found = state_->bindings_.find(key_);
  if (found == state_->bindings_.end() || found->second.version != version_) {
    return;
  }
  auto& binding = found->second;
  if (failed) {
    state_->erase(key_);
    return;
  }
  CHECK_GT(binding.active, 0);
  if (--binding.active != 0) {
    return;
  }
  if (!binding.committed) {
    state_->erase(key_);
    return;
  }
  binding.idle_since = state_->clock_();
  state_->idle_.emplace_back(key_);
  binding.idle_position = std::prev(state_->idle_.end());
}

SessionAwarePolicy::SessionAwarePolicy(
    std::shared_ptr<InstanceMgr> instance_mgr,
    std::unique_ptr<LoadBalancePolicy> fallback,
    int32_t ttl_s,
    int32_t max_entries,
    Clock clock)
    : LoadBalancePolicy(std::move(instance_mgr)),
      fallback_(std::move(fallback)),
      state_(std::make_shared<SessionAffinityState>(ttl_s,
                                                    max_entries,
                                                    std::move(clock))) {
  CHECK(fallback_ != nullptr);
}

bool SessionAwarePolicy::select_instances_pair(
    std::shared_ptr<Request> request) {
  const auto session_id = absl::StripAsciiWhitespace(request->session_id);
  if (session_id.empty()) {
    return fallback_->select_instances_pair(std::move(request));
  }
  // Length framing prevents model/session delimiter collisions.
  std::string key = std::to_string(request->model.size()) + ":";
  key.reserve(key.size() + request->model.size() + session_id.size());
  key.append(request->model);
  key.append(session_id.data(), session_id.size());

  // Expensive fallback work (including CAR prefix lookup) runs without the
  // session mutex. Re-read the binding afterward so concurrent initial turns
  // adopt the first placement that acquired a lease.
  bool placement_attempted = false;
  std::optional<Routing> proposal;
  for (;;) {
    auto candidates =
        instance_mgr_->get_routing_candidates(/*require_load=*/false);
    std::unique_lock<std::mutex> lock(state_->mutex_);
    state_->expire(state_->clock_());
    auto found = state_->bindings_.find(key);
    const Binding* previous =
        found == state_->bindings_.end() ? nullptr : &found->second;
    const RoutingCandidate* prefill =
        previous == nullptr
            ? nullptr
            : bound_candidate(candidates.prefill, previous->prefill);
    const RoutingCandidate* decode =
        previous == nullptr || !previous->decode
            ? nullptr
            : bound_candidate(candidates.decode, *previous->decode);
    const bool complete =
        prefill != nullptr &&
        (decode != nullptr ||
         (!previous->decode && prefill->type == InstanceType::DEFAULT));
    if (!complete && !placement_attempted) {
      lock.unlock();
      placement_attempted = true;
      if (fallback_->select_instances_pair(request)) {
        proposal = request->routing;
      }
      continue;
    }
    if (!complete) {
      if (!proposal.has_value()) {
        return false;
      }
      if (decode == nullptr) {
        decode = find_candidate(candidates.decode, proposal->decode_name);
      }
      if (prefill == nullptr ||
          (decode == nullptr && prefill->type != InstanceType::DEFAULT)) {
        prefill = find_candidate(candidates.prefill, proposal->prefill_name);
      }
    }

    if (!set_route(request, *instance_mgr_, prefill, decode)) {
      return false;
    }

    if (previous == nullptr) {
      if (state_->bindings_.size() >= state_->max_entries_) {
        if (state_->idle_.empty()) {
          LOG(ERROR)
              << "Session affinity capacity is occupied by active requests";
          return false;
        }
        state_->erase(state_->idle_.front());
      }
      found = state_->bindings_.emplace(key, Binding{}).first;
    }
    auto& binding = found->second;
    if (binding.idle_position.has_value()) {
      state_->idle_.erase(*binding.idle_position);
      binding.idle_position.reset();
    }
    const bool unchanged =
        same_instance(binding.prefill, *prefill) &&
        binding.decode.has_value() == (decode != nullptr) &&
        (decode == nullptr || same_instance(*binding.decode, *decode));
    if (!unchanged) {
      binding.prefill = *prefill;
      binding.decode = decode == nullptr
                           ? std::nullopt
                           : std::optional<RoutingCandidate>(*decode);
      binding.version = ++state_->next_version_;
      binding.active = 0;
      binding.committed = false;
    }
    ++binding.active;
    request->session_lease = std::shared_ptr<SessionLease>(
        new SessionLease(state_, std::move(key), binding.version));
    return true;
  }
}

}  // namespace xllm_service
