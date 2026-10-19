include(grpc_proto_library)
include(proto_library)

# A paired engine checkout can supply updated public schemas before its changes
# are committed to the xLLM submodule.
set(XLLM_PROTO_DIR "${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto"
    CACHE PATH "Directory containing the paired xLLM public protobuf schemas")
set(Protobuf_IMPORT_DIRS ${XLLM_PROTO_DIR})

proto_library(
  NAME
    proto_rpc_service
  SRCS
    xllm_rpc_service.proto
)

proto_library(
  NAME
    proto_http_service
  SRCS
    xllm_http_service.proto
)

proto_library(
  NAME
    proto_xllm
  SRCS
    ${XLLM_PROTO_DIR}/anthropic.proto
    ${XLLM_PROTO_DIR}/chat.proto
    ${XLLM_PROTO_DIR}/common.proto
    ${XLLM_PROTO_DIR}/completion.proto
    ${XLLM_PROTO_DIR}/sample.proto
    ${XLLM_PROTO_DIR}/text_generation.proto
    ${XLLM_PROTO_DIR}/rec.proto
    ${XLLM_PROTO_DIR}/disagg_pd.proto
    ${XLLM_PROTO_DIR}/xllm_service.proto
    ${XLLM_PROTO_DIR}/embedding.proto
    ${XLLM_PROTO_DIR}/embedding_data.proto
    ${XLLM_PROTO_DIR}/tensor.proto
    ${XLLM_PROTO_DIR}/multimodal.proto
    ${XLLM_PROTO_DIR}/image_generation.proto
    ${XLLM_PROTO_DIR}/audio_generation.proto
    ${XLLM_PROTO_DIR}/video_generation.proto
    ${XLLM_PROTO_DIR}/rerank.proto
    ${XLLM_PROTO_DIR}/models.proto
)

# RPC structured output reuses the public ChatResponse wire schema.
target_link_libraries(proto_rpc_service PUBLIC proto_xllm)
