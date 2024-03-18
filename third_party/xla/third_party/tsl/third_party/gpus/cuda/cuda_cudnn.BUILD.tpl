licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])
%{multiline_comment}
cc_import(
    name = "cudnn_ops_infer",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_ops_infer.so.%{libcudnn_ops_infer_version}",
)

cc_import(
    name = "cudnn_cnn_infer",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_cnn_infer.so.%{libcudnn_cnn_infer_version}",
)

cc_import(
    name = "cudnn_ops_train",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_ops_train.so.%{libcudnn_ops_train_version}",
)

cc_import(
    name = "cudnn_cnn_train",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_cnn_train.so.%{libcudnn_cnn_train_version}",
)

cc_import(
    name = "cudnn_adv_infer",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_adv_infer.so.%{libcudnn_adv_infer_version}",
)

cc_import(
    name = "cudnn_adv_train",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_adv_train.so.%{libcudnn_adv_train_version}",
)

cc_import(
    name = "cudnn_main",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn.so.%{libcudnn_version}",
)
%{multiline_comment}
cc_library(
    name = "cudnn",
    %{multiline_comment}
    deps = [
      ":cudnn_ops_infer",
      ":cudnn_ops_train",
      ":cudnn_cnn_infer",
      ":cudnn_cnn_train",
      ":cudnn_adv_infer",
      ":cudnn_adv_train",
      "@cuda_nvrtc//:nvrtc",
      ":cudnn_main",
    ],
    %{multiline_comment}
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**",
    ]),
    include_prefix = "third_party/gpus/cudnn",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
