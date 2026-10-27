// Copyright 2026 The ML Drift Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift/common/task/serialization_base.h"

#include <vector>

#include "gtest/gtest.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/verifier.h"
#include "ml_drift/cl/serialization_generated.h"
#include "ml_drift/common/task/serialization_base_generated.h"
#include "ml_drift/common/types.h"
#include "ml_drift/metal/inference_context_generated.h"
#include "ml_drift/webgpu/serialization_generated.h"

namespace ml_drift {
namespace {

// One Int3 table per node gives more tables than the default limit of
// flatbuffers::Verifier (1,000,000).
constexpr int kNumNodes = 1100000;

template <typename InferenceContextBuilder, typename VerifyBuffer>
void ExpectVerifiedOnlyWithScaledLimit(VerifyBuffer verify_buffer) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<data::Int3>> work_group_sizes(kNumNodes);
  for (auto& work_group_size : work_group_sizes) {
    work_group_size = Encode(int3(8, 4, 1), &builder);
  }
  auto work_group_sizes_fb = builder.CreateVector(work_group_sizes);
  InferenceContextBuilder inference_context_builder(builder);
  inference_context_builder.add_tuned_work_group_sizes_per_node(
      work_group_sizes_fb);
  builder.Finish(inference_context_builder.Finish());

  flatbuffers::Verifier default_verifier(builder.GetBufferPointer(),
                                         builder.GetSize());
  EXPECT_FALSE(verify_buffer(default_verifier));
  flatbuffers::Verifier verifier(
      builder.GetBufferPointer(), builder.GetSize(),
      GetInferenceContextVerifierOptions(builder.GetSize()));
  EXPECT_TRUE(verify_buffer(verifier));
}

TEST(GetInferenceContextVerifierOptionsTest, AcceptsLargeClContext) {
  ExpectVerifiedOnlyWithScaledLimit<cl::data::InferenceContextBuilder>(
      [](auto& v) { return cl::data::VerifyInferenceContextBuffer(v); });
}

TEST(GetInferenceContextVerifierOptionsTest, AcceptsLargeMetalContext) {
  ExpectVerifiedOnlyWithScaledLimit<metal::data::InferenceContextBuilder>(
      [](auto& v) { return metal::data::VerifyInferenceContextBuffer(v); });
}

TEST(GetInferenceContextVerifierOptionsTest, AcceptsLargeWebGpuContext) {
  ExpectVerifiedOnlyWithScaledLimit<webgpu::data::InferenceContextBuilder>(
      [](auto& v) { return webgpu::data::VerifyInferenceContextBuffer(v); });
}

}  // namespace
}  // namespace ml_drift
