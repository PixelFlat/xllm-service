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

#include "common/call_data.h"

#include <exception>
#include <nlohmann/json.hpp>
#include <string>

namespace xllm_service {
namespace {

bool omit_empty_tool_calls(std::string* response_json, std::string* error) {
  if (response_json->find("\"tool_calls\":[]") == std::string::npos) {
    return true;
  }

  try {
    nlohmann::ordered_json response =
        nlohmann::ordered_json::parse(*response_json);
    auto choices = response.find("choices");
    if (choices == response.end() || !choices->is_array()) {
      return true;
    }

    for (auto& choice : *choices) {
      if (!choice.is_object()) {
        continue;
      }
      for (const char* message_key : {"message", "delta"}) {
        auto message = choice.find(message_key);
        if (message == choice.end() || !message->is_object()) {
          continue;
        }
        auto tool_calls = message->find("tool_calls");
        if (tool_calls != message->end() && tool_calls->is_array() &&
            tool_calls->empty()) {
          message->erase(tool_calls);
        }
      }
    }

    *response_json = response.dump();
    return true;
  } catch (const nlohmann::json::exception& e) {
    *error = "Failed to remove empty tool calls: " + std::string(e.what());
    return false;
  } catch (const std::exception& e) {
    *error = "Failed to remove empty tool calls: " + std::string(e.what());
    return false;
  }
}

}  // namespace

std::pair<int32_t, std::string> serialize_http_error(const llm::Status& status,
                                                     bool anthropic) {
  CHECK(!status.ok());
  int32_t http_status = 500;
  const char* error_type = "server_error";
  switch (status.code()) {
    case llm::StatusCode::INVALID_ARGUMENT:
      http_status = 400;
      error_type = "invalid_request_error";
      break;
    case llm::StatusCode::RESOURCE_EXHAUSTED:
      http_status = 429;
      error_type = "rate_limit_error";
      break;
    case llm::StatusCode::UNAUTHENTICATED:
      http_status = 401;
      error_type = "authentication_error";
      break;
    case llm::StatusCode::UNAVAILABLE:
      http_status = 503;
      break;
    case llm::StatusCode::DEADLINE_EXCEEDED:
      http_status = 504;
      break;
    case llm::StatusCode::UNIMPLEMENTED:
      http_status = 501;
      break;
    case llm::StatusCode::CANCELLED:
      http_status = 499;
      break;
    default:
      break;
  }
  nlohmann::json response = {
      {"error", {{"message", status.message()}, {"type", error_type}}}};
  if (anthropic) {
    response["type"] = "error";
    if (http_status >= 500) {
      response["error"]["type"] = "api_error";
    }
  } else {
    response["error"]["param"] = nullptr;
    response["error"]["code"] = http_status;
  }
  return {http_status, response.dump()};
}

bool ChatCallData::write_and_finish(
    const ::xllm::proto::ChatResponse& response) {
  std::string response_json;
  std::string error;
  if (!serialize_response(response, &response_json, &error) ||
      !omit_empty_tool_calls(&response_json, &error)) {
    return finish_with_error(error);
  }
  return Base::write_and_finish(response_json);
}

bool ChatCallData::write(const ::xllm::proto::ChatResponse& response) {
  std::string response_json;
  std::string error;
  if (!serialize_response(response, &response_json, &error) ||
      !omit_empty_tool_calls(&response_json, &error)) {
    LOG(ERROR) << error;
    return false;
  }
  return Base::write("data: " + response_json + "\n\n");
}

}  // namespace xllm_service
