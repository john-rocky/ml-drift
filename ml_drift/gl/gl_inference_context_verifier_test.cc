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

#include <vector>

#include "gtest/gtest.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/verifier.h"
#include "ml_drift/common/task/serialization_base.h"
#include "ml_drift/common/task/serialization_base_generated.h"
#include "ml_drift/common/types.h"
#include "ml_drift/gl/gl_inference_context_generated.h"

namespace ml_drift {
namespace gl {
namespace {

// One Int3 table per node gives more tables than the default limit of
// flatbuffers::Verifier (1,000,000). The other backends' schemas are checked
// in ml_drift/common/task/serialization_base_test.cc.
constexpr int kNumNodes = 1100000;

TEST(GlInferenceContextVerifierTest, AcceptsLargeContext) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<ml_drift::data::Int3>> work_group_sizes(
      kNumNodes);
  for (auto& work_group_size : work_group_sizes) {
    work_group_size = ml_drift::Encode(int3(8, 4, 1), &builder);
  }
  auto work_group_sizes_fb = builder.CreateVector(work_group_sizes);
  data::InferenceContextBuilder inference_context_builder(builder);
  inference_context_builder.add_tuned_work_group_sizes_per_node(
      work_group_sizes_fb);
  builder.Finish(inference_context_builder.Finish());

  flatbuffers::Verifier default_verifier(builder.GetBufferPointer(),
                                         builder.GetSize());
  EXPECT_FALSE(data::VerifyInferenceContextBuffer(default_verifier));
  flatbuffers::Verifier verifier(
      builder.GetBufferPointer(), builder.GetSize(),
      GetInferenceContextVerifierOptions(builder.GetSize()));
  EXPECT_TRUE(data::VerifyInferenceContextBuffer(verifier));
}

}  // namespace
}  // namespace gl
}  // namespace ml_drift
