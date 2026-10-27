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

#include "ml_drift/common/selectors/operation_selector.h"

#include <any>
#include <cstdint>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_util.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/ir_model_util.h"
#include "ml_drift/common/kernels/fully_connected.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace {

using ::testing::Contains;
using ::testing::ElementsAreArray;
using ::testing::Not;
using ::testing::TestParamInfo;
using ::testing::Values;

constexpr int kSrcChannels = 64;
constexpr int kDstChannels = 32;
constexpr int kBlockSize = 32;

GpuInfo GetTestGpuInfo() {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  gpu_info.vendor = GpuVendor::kApple;
  return gpu_info;
}

CreateGpuModelInfo GetTestCreateInfo() {
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;
  return create_info;
}

WeightsDescription GetWeightsDesc(const GpuInfo& gpu_info,
                                  OperationType op_type, const OHWI& shape) {
  switch (op_type) {
    case OperationType::kFullyConnectedInt2:
      return GetFullyConnectedInt2WeightsDesc(gpu_info, shape);
    case OperationType::kFullyConnectedInt4:
      return GetFullyConnectedInt4WeightsDesc(gpu_info, shape);
    default:
      return GetFullyConnectedInt8WeightsDesc(gpu_info, shape);
  }
}

Tensor<OHWI, DataType::kFloat32> MakeBlockwiseScale(float value) {
  Tensor<OHWI, DataType::kFloat32> scale;
  scale.shape = OHWI(kDstChannels, 1, 1, kSrcChannels / kBlockSize);
  scale.data.assign(scale.shape.DimensionsProduct(), value);
  return scale;
}

// Attributes of a [kDstChannels, kSrcChannels] FullyConnected with blockwise
// scales. They also hold weights and scales, so the node is valid with constant
// weights too; the tests tell the two paths apart by the node inputs.
std::any MakeAttributes(OperationType op_type) {
  Tensor<OHWI, DataType::kInt8> weights;
  weights.shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  weights.data.resize(weights.shape.DimensionsProduct());
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = i % 4 - 2;  // In the int2 range.
  }
  const Tensor<OHWI, DataType::kFloat32> scale = MakeBlockwiseScale(0.01f);
  switch (op_type) {
    case OperationType::kFullyConnectedInt2: {
      FullyConnectedInt2Attributes attr;
      attr.weights = weights;
      attr.scale = scale;
      return attr;
    }
    case OperationType::kFullyConnectedInt4: {
      FullyConnectedInt4Attributes attr;
      attr.weights = weights;
      attr.scale = scale;
      return attr;
    }
    default: {
      FullyConnectedInt8Attributes attr;
      attr.weights = weights;
      attr.scale = scale;
      return attr;
    }
  }
}

// Descriptors of the weights, the scale and, optionally, the zero point that a
// delegate passes to a FullyConnected as inputs (external weights).
std::vector<TensorDescriptor> MakeExternalWeightsInputs(const GpuInfo& gpu_info,
                                                        OperationType op_type,
                                                        bool with_zero_point) {
  const OHWI weights_shape(kDstChannels, 1, 1, kSrcChannels);
  std::vector<TensorDescriptor> inputs = {GetTensorDescriptorsForWeightsLayout(
      weights_shape, GetWeightsDesc(gpu_info, op_type, weights_shape))[0]};
  inputs.push_back(ScaleOrZeroPointToTensorDesc(
      gpu_info, MakeBlockwiseScale(0.01f), DataType::kFloat32));
  if (with_zero_point) {
    inputs.push_back(ScaleOrZeroPointToTensorDesc(
        gpu_info, MakeBlockwiseScale(0.0f), DataType::kFloat32));
  }
  return inputs;
}

// Converts a GraphFloat32 with one FullyConnected node of `op_type` whose
// inputs are src and `external_inputs` (predefined graph inputs). Returns the
// ids of the node inputs in `fc_inputs`.
absl::Status ConvertGraph(const GpuInfo& gpu_info, OperationType op_type,
                          const std::vector<TensorDescriptor>& external_inputs,
                          GpuModel* gpu_model,
                          std::vector<ValueId>* fc_inputs) {
  CreateGpuModelInfo create_info = GetTestCreateInfo();
  GraphFloat32 graph;
  Value* src = graph.NewValue();
  src->tensor.type = DataType::kFloat32;
  src->tensor.shape = BHWC(1, 1, 1, kSrcChannels);
  Node* fc = graph.NewNode();
  fc->operation.type = ToString(op_type);
  fc->operation.attributes = MakeAttributes(op_type);
  graph.AddConsumer(fc->id, src->id);
  fc_inputs->push_back(src->id);
  for (const TensorDescriptor& desc : external_inputs) {
    Value* value = graph.NewValue();
    value->tensor.type = desc.GetDataType();
    value->tensor.shape = desc.GetBHWCShape();
    graph.AddConsumer(fc->id, value->id);
    create_info.predefined[value->id] = desc;
    fc_inputs->push_back(value->id);
  }
  Value* dst = nullptr;
  ABSL_RETURN_IF_ERROR(AddOutput(&graph, fc, &dst));
  dst->tensor.type = DataType::kFloat32;
  dst->tensor.shape = BHWC(1, 1, 1, kDstChannels);
  return GraphToGpuModel(graph, create_info, gpu_info, gpu_model);
}

// The same model as ConvertGraph, as an IrModel.
absl::Status ConvertIrModel(
    const GpuInfo& gpu_info, OperationType op_type,
    const std::vector<TensorDescriptor>& external_inputs, GpuModel* gpu_model,
    std::vector<ValueId>* fc_inputs) {
  CreateGpuModelInfo create_info = GetTestCreateInfo();
  ir::IrModel ir_model;
  ir::IrTensor* src =
      ir_model.add_tensor(DataType::kFloat32, BHWC(1, 1, 1, kSrcChannels));
  ir_model.add_input(src->id);
  ir::IrOp* fc = ir_model.add_op();
  fc->name = ToString(op_type);
  fc->attr = MakeAttributes(op_type);
  ir_model.AddConsumer(src->id, fc->id);
  fc_inputs->push_back(static_cast<ValueId>(src->id));
  for (const TensorDescriptor& desc : external_inputs) {
    ir::IrTensor* tensor =
        ir_model.add_tensor(desc.GetDataType(), desc.GetBHWCShape());
    ir_model.add_input(tensor->id);
    ir_model.AddConsumer(tensor->id, fc->id);
    create_info.predefined[static_cast<ValueId>(tensor->id)] = desc;
    fc_inputs->push_back(static_cast<ValueId>(tensor->id));
  }
  ir::IrTensor* dst =
      ir_model.add_tensor(DataType::kFloat32, BHWC(1, 1, 1, kDstChannels));
  ir_model.SetProducer(dst->id, fc->id);
  ir_model.add_output(dst->id);
  return ir::IrModelToGpuModel(ir_model, create_info, gpu_info, gpu_model);
}

// Expects one FullyConnected node that reads the external inputs directly,
// with or without a zero-point input.
void ExpectExternalWeightsFullyConnected(const GpuModel& gpu_model,
                                         const std::vector<ValueId>& fc_inputs,
                                         bool reads_zero_point) {
  ASSERT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_THAT(gpu_model.nodes[0].inputs, ElementsAreArray(fc_inputs));
  const std::vector<std::string>& src_names =
      gpu_model.nodes[0].gpu_operation->GetSrcTensorsNames();
  if (reads_zero_point) {
    EXPECT_THAT(src_names, Contains("weights_zero_point"));
  } else {
    EXPECT_THAT(src_names, Not(Contains("weights_zero_point")));
  }
}

class QuantizedFullyConnectedExternalWeightsTest
    : public ::testing::TestWithParam<OperationType> {};

// src, weights and scale: a FullyConnected without zero point.
TEST_P(QuantizedFullyConnectedExternalWeightsTest, GraphWithoutZeroPoint) {
  const GpuInfo gpu_info = GetTestGpuInfo();
  GpuModel gpu_model;
  std::vector<ValueId> fc_inputs;
  ABSL_ASSERT_OK(ConvertGraph(
      gpu_info, GetParam(),
      MakeExternalWeightsInputs(gpu_info, GetParam(),
                                /*with_zero_point=*/false),
      &gpu_model, &fc_inputs));
  ExpectExternalWeightsFullyConnected(gpu_model, fc_inputs,
                                      /*reads_zero_point=*/false);
}

// src, weights, scale and zero point.
TEST_P(QuantizedFullyConnectedExternalWeightsTest, GraphWithZeroPoint) {
  const GpuInfo gpu_info = GetTestGpuInfo();
  GpuModel gpu_model;
  std::vector<ValueId> fc_inputs;
  ABSL_ASSERT_OK(ConvertGraph(
      gpu_info, GetParam(),
      MakeExternalWeightsInputs(gpu_info, GetParam(),
                                /*with_zero_point=*/true),
      &gpu_model, &fc_inputs));
  ExpectExternalWeightsFullyConnected(gpu_model, fc_inputs,
                                      /*reads_zero_point=*/true);
}

TEST_P(QuantizedFullyConnectedExternalWeightsTest, IrModelWithoutZeroPoint) {
  const GpuInfo gpu_info = GetTestGpuInfo();
  GpuModel gpu_model;
  std::vector<ValueId> fc_inputs;
  ABSL_ASSERT_OK(ConvertIrModel(
      gpu_info, GetParam(),
      MakeExternalWeightsInputs(gpu_info, GetParam(),
                                /*with_zero_point=*/false),
      &gpu_model, &fc_inputs));
  ExpectExternalWeightsFullyConnected(gpu_model, fc_inputs,
                                      /*reads_zero_point=*/false);
}

TEST_P(QuantizedFullyConnectedExternalWeightsTest, IrModelWithZeroPoint) {
  const GpuInfo gpu_info = GetTestGpuInfo();
  GpuModel gpu_model;
  std::vector<ValueId> fc_inputs;
  ABSL_ASSERT_OK(ConvertIrModel(
      gpu_info, GetParam(),
      MakeExternalWeightsInputs(gpu_info, GetParam(),
                                /*with_zero_point=*/true),
      &gpu_model, &fc_inputs));
  ExpectExternalWeightsFullyConnected(gpu_model, fc_inputs,
                                      /*reads_zero_point=*/true);
}

INSTANTIATE_TEST_SUITE_P(
    OperationSelectorTest, QuantizedFullyConnectedExternalWeightsTest,
    Values(OperationType::kFullyConnectedInt2,
           OperationType::kFullyConnectedInt4,
           OperationType::kFullyConnectedInt8),
    [](const TestParamInfo<OperationType>& info) {
      return ToString(info.param);
    });

}  // namespace
}  // namespace ml_drift
