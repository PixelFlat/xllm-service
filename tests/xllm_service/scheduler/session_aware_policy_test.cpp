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

#include <absl/time/clock.h>
#include <brpc/server.h>
#include <gtest/gtest.h>

#include <future>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <utility>

#include "http_service/service.h"
#include "scheduler/loadbalance_policy/cache_aware_routing.h"
#include "scheduler/loadbalance_policy/round_robin.h"
#include "scheduler/scheduler.h"

namespace xllm_service {

class SchedulerTestPeer final {
 public:
  static std::shared_ptr<InstanceMgr> manager() {
    return std::shared_ptr<InstanceMgr>(new InstanceMgr());
  }

  static void instances(InstanceMgr& manager,
                        std::vector<InstanceMetaInfo> values) {
    std::unique_lock<std::shared_mutex> lock(manager.cluster_mutex_);
    manager.instances_.clear();
    manager.prefill_index_.clear();
    manager.decode_index_.clear();
    manager.suspect_instances_.clear();
    for (auto& value : values) {
      const std::string name = value.name;
      manager.add_instance_to_index(name, value);
      auto inserted = manager.instances_.emplace(name, std::move(value));
      if (inserted.first->second.runtime_state ==
          InstanceRuntimeState::SUSPECT) {
        manager.mark_instance_suspect(name,
                                      inserted.first->second.incarnation_id);
      }
    }
  }

  static std::unique_ptr<Scheduler> scheduler(
      std::unique_ptr<LoadBalancePolicy> policy,
      std::shared_ptr<InstanceMgr> manager = nullptr) {
    auto scheduler = std::unique_ptr<Scheduler>(new Scheduler());
    scheduler->options_.default_backend_type("vllm");
    scheduler->lb_policy_ = std::move(policy);
    scheduler->instance_mgr_ = std::move(manager);
    return scheduler;
  }

  static bool publish(Scheduler& scheduler, std::shared_ptr<Request> value) {
    return scheduler.publish_request(std::move(value));
  }

  static void drain(Scheduler& scheduler) {
    for (auto& pool : scheduler.output_threadpools_) {
      std::promise<void> done;
      auto completed = done.get_future();
      pool.schedule([&done] { done.set_value(); });
      completed.wait();
    }
  }
};

namespace {

InstanceMetaInfo instance(std::string name,
                          InstanceType type = InstanceType::DEFAULT,
                          std::string incarnation = "original") {
  InstanceMetaInfo info;
  info.name = std::move(name);
  info.type = type;
  info.incarnation_id = std::move(incarnation);
  info.register_ts_ms = 1;
  info.dp_size = 1;
  info.kv_split_size = 1;
  return info;
}

std::shared_ptr<Request> request(std::string session,
                                 std::string model = "model") {
  auto result = std::make_shared<Request>();
  result->session_id = std::move(session);
  result->model = std::move(model);
  return result;
}

class SessionAwarePolicyTest : public testing::Test {
 protected:
  void SetUp() override {
    manager_ = SchedulerTestPeer::manager();
    SchedulerTestPeer::instances(*manager_, {instance("A"), instance("B")});
    policy_ = make_policy(/*capacity=*/64);
  }

  std::unique_ptr<SessionAwarePolicy> make_policy(int32_t capacity) {
    return std::make_unique<SessionAwarePolicy>(
        manager_,
        std::make_unique<RoundRobin>(manager_),
        /*ttl_s=*/300,
        capacity,
        [this] { return now_; });
  }

  std::shared_ptr<Request> route(std::string session,
                                 std::string model = "model") {
    auto value = request(std::move(session), std::move(model));
    if (!policy_->select_instances_pair(value)) {
      return nullptr;
    }
    EXPECT_TRUE(manager_->bind_request_instance_incarnations(value));
    return value;
  }

  void complete(const std::shared_ptr<Request>& value) {
    ASSERT_NE(value, nullptr);
    ASSERT_NE(value->session_lease, nullptr);
    value->session_lease->commit();
    value->session_lease->finish(/*failed=*/false);
  }

  SessionAwarePolicy::TimePoint now_{};
  std::shared_ptr<InstanceMgr> manager_;
  std::unique_ptr<SessionAwarePolicy> policy_;
};

TEST_F(SessionAwarePolicyTest, IdentityFallbackAndModelIsolation) {
  auto first = route("");
  auto second = route("  ");
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_NE(first->routing.prefill_name, second->routing.prefill_name);
  EXPECT_EQ(first->session_lease, nullptr);
  auto session = route("conversation");
  ASSERT_NO_FATAL_FAILURE(complete(session));
  auto other_model = route("conversation", "another-model");
  ASSERT_NE(other_model, nullptr);
  EXPECT_NE(other_model->routing.prefill_name, session->routing.prefill_name);
  auto repeated = route(" conversation ");
  ASSERT_NE(repeated, nullptr);
  EXPECT_EQ(repeated->routing.prefill_name, session->routing.prefill_name);
}

TEST_F(SessionAwarePolicyTest, ConcurrentInitialPlacement) {
  std::vector<std::future<std::shared_ptr<Request>>> pending;
  pending.reserve(8);
  for (int32_t i = 0; i < 8; ++i) {
    pending.emplace_back(
        std::async(std::launch::async, [this] { return route("concurrent"); }));
  }
  std::vector<std::shared_ptr<Request>> active;
  active.reserve(pending.size());
  for (auto& result : pending) {
    active.emplace_back(result.get());
    ASSERT_NE(active.back(), nullptr);
    EXPECT_EQ(active.back()->routing.prefill_name, "A");
  }
  // No dispatch was confirmed: dropping the last hold must not commit a pin.
  active.clear();
  auto retry = route("concurrent");
  ASSERT_NE(retry, nullptr);
  EXPECT_EQ(retry->routing.prefill_name, "B");
}

TEST_F(SessionAwarePolicyTest, ActiveRequestsDelayExpiry) {
  auto first = route("active");
  ASSERT_NE(first, nullptr);
  first->session_lease->commit();
  now_ += std::chrono::seconds(1000);
  auto overlapping = route("active");
  ASSERT_NE(overlapping, nullptr);
  EXPECT_EQ(overlapping->routing.prefill_name, first->routing.prefill_name);
  first->session_lease->finish(/*failed=*/false);
  now_ += std::chrono::seconds(1000);
  auto still_active = route("active");
  ASSERT_NE(still_active, nullptr);
  EXPECT_EQ(still_active->routing.prefill_name, first->routing.prefill_name);
  ASSERT_NO_FATAL_FAILURE(complete(overlapping));
  ASSERT_NO_FATAL_FAILURE(complete(still_active));
  now_ += std::chrono::seconds(300);
  auto expired = route("active");
  ASSERT_NE(expired, nullptr);
  EXPECT_NE(expired->routing.prefill_name, first->routing.prefill_name);
}

TEST_F(SessionAwarePolicyTest, ReplacementFencesOldCompletions) {
  auto old = route("restart");
  ASSERT_NE(old, nullptr);
  old->session_lease->commit();
  SchedulerTestPeer::instances(
      *manager_,
      {instance("A", InstanceType::DEFAULT, "restarted"), instance("B")});
  // A second binding pass must reject the old snapshot, not silently rebind it.
  EXPECT_FALSE(manager_->bind_request_instance_incarnations(old));
  auto replacement = route("restart");
  ASSERT_NE(replacement, nullptr);
  EXPECT_EQ(replacement->routing.prefill_name, "B");
  ASSERT_NO_FATAL_FAILURE(complete(replacement));
  old->session_lease->finish(/*failed=*/true);
  auto repeated = route("restart");
  ASSERT_NE(repeated, nullptr);
  EXPECT_EQ(repeated->routing.prefill_name, "B");
  repeated->session_lease->finish(/*failed=*/true);
  auto after_error = route("restart");
  ASSERT_NE(after_error, nullptr);
  EXPECT_EQ(after_error->routing.prefill_name, "A");
}

TEST_F(SessionAwarePolicyTest, PdFailoverRetainsHealthyPeer) {
  auto p0 = instance("P0", InstanceType::PREFILL);
  auto p1 = instance("P1", InstanceType::PREFILL);
  auto d0 = instance("D0", InstanceType::DECODE);
  auto d1 = instance("D1", InstanceType::DECODE);
  SchedulerTestPeer::instances(*manager_, {p0, p1, d0, d1});
  auto first = route("pd");
  ASSERT_NO_FATAL_FAILURE(complete(first));
  EXPECT_EQ(first->routing.prefill_name, "P0");
  EXPECT_EQ(first->routing.decode_name, "D0");
  p0.incarnation_id = "new-prefill";
  SchedulerTestPeer::instances(*manager_, {p0, p1, d0, d1});
  auto p_changed = route("pd");
  ASSERT_NO_FATAL_FAILURE(complete(p_changed));
  EXPECT_EQ(p_changed->routing.prefill_name, "P1");
  EXPECT_EQ(p_changed->routing.decode_name, "D0");
  d0.runtime_state = InstanceRuntimeState::SUSPECT;
  SchedulerTestPeer::instances(*manager_, {p0, p1, d0, d1});
  auto d_changed = route("pd");
  ASSERT_NO_FATAL_FAILURE(complete(d_changed));
  EXPECT_EQ(d_changed->routing.prefill_name, "P1");
  EXPECT_EQ(d_changed->routing.decode_name, "D1");
  SchedulerTestPeer::instances(*manager_, {p0, p1});
  EXPECT_EQ(route("pd"), nullptr);
}

TEST_F(SessionAwarePolicyTest, CapacityProtectsActiveSessions) {
  policy_ = make_policy(/*capacity=*/2);
  auto first = route("first");
  auto second = route("second");
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(route("third"), nullptr);
  ASSERT_NO_FATAL_FAILURE(complete(first));
  auto third = route("third");
  ASSERT_NO_FATAL_FAILURE(complete(third));
  auto repeated_second = route("second");
  ASSERT_NE(repeated_second, nullptr);
  EXPECT_EQ(repeated_second->routing.prefill_name,
            second->routing.prefill_name);
  ASSERT_NO_FATAL_FAILURE(complete(second));
  ASSERT_NO_FATAL_FAILURE(complete(repeated_second));
  // Touch third, making second the least recently used idle binding.
  auto touched = route("third");
  ASSERT_NO_FATAL_FAILURE(complete(touched));
  auto fourth = route("fourth");
  ASSERT_NE(fourth, nullptr);
  auto same_third = route("third");
  ASSERT_NE(same_third, nullptr);
  EXPECT_EQ(same_third->routing.prefill_name, third->routing.prefill_name);
}

TEST_F(SessionAwarePolicyTest, HardBindingIgnoresLoadChanges) {
  const auto load = [this](const std::string& name, uint64_t waiting) {
    proto::LoadMetrics metrics;
    metrics.set_waiting_requests_num(waiting);
    metrics.set_gpu_cache_usage_perc(0.1);
    manager_->record_load_metrics_update(name, metrics);
  };
  load("A", 10);
  load("B", 0);
  policy_ = std::make_unique<SessionAwarePolicy>(
      manager_,
      std::make_unique<CacheAwareRouting>(manager_, nullptr),
      /*ttl_s=*/300,
      /*max_entries=*/64,
      [this] { return now_; });
  auto first = route("car");
  ASSERT_NO_FATAL_FAILURE(complete(first));
  EXPECT_EQ(first->routing.prefill_name, "B");
  load("A", 0);
  load("B", 10);
  auto repeated = route("car");
  ASSERT_NE(repeated, nullptr);
  EXPECT_EQ(repeated->routing.prefill_name, "B");
  auto cold = route("cold");
  ASSERT_NE(cold, nullptr);
  EXPECT_EQ(cold->routing.prefill_name, "A");
}

class ConnectedCall final : public CallData {
 public:
  bool proceed(bool /*rpc_ok*/) override { return true; }
  bool is_disconnected() const override { return false; }
};

TEST_F(SessionAwarePolicyTest, OutputBeforeAckAndCancellation) {
  auto* policy = policy_.get();
  auto scheduler = SchedulerTestPeer::scheduler(std::move(policy_), manager_);
  const auto generate = [&](std::string request_id, bool cancelled) {
    auto value = request("decode-output");
    value->service_request_id = std::move(request_id);
    value->call_data = std::make_shared<ConnectedCall>();
    value->latest_generate_time = absl::Now();
    value->output_callback = [](const llm::RequestOutput&) { return true; };
    EXPECT_TRUE(policy->select_instances_pair(value));
    EXPECT_TRUE(SchedulerTestPeer::publish(*scheduler, value));
    llm::RequestOutput output;
    output.service_request_id = value->service_request_id;
    output.finished = true;
    if (cancelled) {
      output.status =
          llm::Status(llm::StatusCode::CANCELLED, "client cancelled");
    }
    EXPECT_TRUE(scheduler->handle_generation(std::move(output)));
    SchedulerTestPeer::drain(*scheduler);
    return value;
  };
  auto first = generate("first", /*cancelled=*/false);
  auto cancelled = generate("cancelled", /*cancelled=*/true);
  EXPECT_EQ(first->routing.prefill_name, cancelled->routing.prefill_name);
  auto repeated = request("decode-output");
  ASSERT_TRUE(policy->select_instances_pair(repeated));
  EXPECT_EQ(repeated->routing.prefill_name, first->routing.prefill_name);
}

class CaptureSessionPolicy final : public LoadBalancePolicy {
 public:
  CaptureSessionPolicy() : LoadBalancePolicy(nullptr) {}
  bool select_instances_pair(std::shared_ptr<Request> value) override {
    std::lock_guard<std::mutex> lock(mutex_);
    observed_ = value->session_id;
    return false;  // Stop before dispatch; exercise HTTP parsing and
                   // scheduling.
  }
  std::string observed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return observed_;
  }

 private:
  mutable std::mutex mutex_;
  std::string observed_;
};

TEST(SessionHeaderTest, ReadsCaseInsensitiveHeaderForAllInferenceEndpoints) {
  auto policy = std::make_unique<CaptureSessionPolicy>();
  auto* captured = policy.get();
  auto scheduler = SchedulerTestPeer::scheduler(std::move(policy));
  Options options;
  options.num_threads(1);
  XllmHttpServiceImpl service(options, scheduler.get());
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service,
                              brpc::SERVER_DOESNT_OWN_SERVICE,
                              "/v1/completions => Completions,"
                              "/v1/chat/completions => ChatCompletions,"
                              "/v1/messages => AnthropicMessages"),
            0);
  brpc::ServerOptions server_options;
  server_options.num_threads = 2;
  ASSERT_EQ(server.Start("127.0.0.1:0", &server_options), 0);
  brpc::Channel channel;
  brpc::ChannelOptions channel_options;
  channel_options.protocol = "http";
  channel_options.timeout_ms = 3000;
  ASSERT_EQ(channel.Init(server.listen_address(), &channel_options), 0);
  const std::vector<std::string> endpoints = {
      "/v1/completions", "/v1/chat/completions", "/v1/messages"};
  for (const auto& endpoint : endpoints) {
    brpc::Controller controller;
    controller.http_request().uri() = endpoint;
    controller.http_request().set_method(brpc::HTTP_METHOD_POST);
    controller.http_request().SetHeader("x-SeSsIoN-Id", endpoint);
    controller.http_request().SetHeader("Content-Type", "application/json");
    controller.request_attachment().append(
        R"({"model":"model","prompt":"hello","max_tokens":8,"messages":[{"role":"user","content":"hello"}]})");
    channel.CallMethod(nullptr, &controller, nullptr, nullptr, nullptr);
    EXPECT_EQ(captured->observed(), endpoint);
  }
}

}  // namespace
}  // namespace xllm_service
