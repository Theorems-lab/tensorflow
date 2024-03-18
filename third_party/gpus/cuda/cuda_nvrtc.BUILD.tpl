licenses(["restricted"])  # NVIDIA proprietary license
%{multiline_comment}
cc_import(
    name = "nvrtc_main",
    hdrs = [":headers"],
    shared_library = "lib/libnvrtc.so.%{libnvrtc_version}",
)

cc_import(
    name = "nvrtc_builtins",
    hdrs = [":headers"],
    shared_library = "lib/libnvrtc-builtins.so.%{libnvrtc-builtins_version}",
)
%{multiline_comment}
cc_library(
    name = "nvrtc",
    %{multiline_comment}
    deps = [
        ":nvrtc_main",
        ":nvrtc_builtins",
    ],
    %{multiline_comment}
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
