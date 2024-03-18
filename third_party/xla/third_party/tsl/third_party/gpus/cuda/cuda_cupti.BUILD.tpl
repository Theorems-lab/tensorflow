licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])
%{multiline_comment}
cc_import(
    name = "cupti_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libcupti.so.%{libcupti_version}",
)
%{multiline_comment}
cc_library(
    name = "cupti",
    %{comment}deps = [":cupti_shared_library"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**",
    ]),
    include_prefix = "third_party/gpus/cuda/extras/CUPTI/include",
    includes = ["include/"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
