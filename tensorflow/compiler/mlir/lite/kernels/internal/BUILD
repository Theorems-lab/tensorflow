load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "compatibility_macros",
    hdrs = ["compatibility_macros.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
)

cc_library(
    name = "portable_tensor_utils",
    srcs = [
        "portable_tensor_utils.cc",
    ],
    hdrs = [
        "portable_tensor_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
)
