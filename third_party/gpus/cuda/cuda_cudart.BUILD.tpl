licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

filegroup(
    name = "static",
    srcs = ["lib/libcudart_static.a"],
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
%{multiline_comment}
cc_import(
    name = "cuda_driver_shared_library",
    shared_library = "lib/stubs/libcuda.so",
)

cc_import(
    name = "cudart_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libcudart.so.%{libcudart_version}",
)
%{multiline_comment}
cc_library(
    name = "cuda_driver",
    %{comment}deps = [":cuda_driver_shared_library"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudart",
    %{comment}deps = [":cudart_shared_library"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**",
    ]),
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
