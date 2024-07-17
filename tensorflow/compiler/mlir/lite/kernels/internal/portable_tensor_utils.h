/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_INTERNAL_PORTABLE_TENSOR_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_INTERNAL_PORTABLE_TENSOR_UTILS_H_

#include <cstdint>

namespace mlirlite {
namespace tensor_utils {

void PortableSymmetricQuantizeFloats(const float* values, int size,
                                     int8_t* quantized_values, float* min_value,
                                     float* max_value, float* scaling_factor);

void PortableSymmetricQuantizeFloats(const float* values, int size,
                                     int8_t* quantized_values, float min_value,
                                     float max_value, float* scaling_factor);

}  // namespace tensor_utils
}  // namespace mlirlite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_INTERNAL_PORTABLE_TENSOR_UTILS_H_
