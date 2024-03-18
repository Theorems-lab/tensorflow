# Copyright 2024 The TensorFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hermetic NCCL repositories initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_mirror_urls")
load(
    "//third_party/gpus/cuda:hermetic_cuda_redist_init_repositories.bzl",
    "OS_ARCH_DICT",
    "create_build_file_and_return_redist_major_version",
    "create_dummy_build_file",
    "get_archive_name",
    "get_env_var",
    "use_local_cuda_path",
)

def _use_downloaded_nccl_wheel(repository_ctx):
    """ Downloads NCCL wheel and inits hermetic NCCL repository."""
    cuda_version = (get_env_var(repository_ctx, "HERMETIC_CUDA_VERSION") or
                    get_env_var(repository_ctx, "TF_CUDA_VERSION"))
    major_version = ""
    major_version_to_build_template_dict = {
        v: k
        for k, v in repository_ctx.attr.build_template_to_major_version_dict.items()
    }
    if cuda_version:
        # Download archive only when GPU config is used.
        arch = OS_ARCH_DICT[repository_ctx.os.arch]
        dict_key = "{cuda_version}-{arch}".format(
            cuda_version = cuda_version,
            arch = arch,
        )
        supported_versions = repository_ctx.attr.url_dict.keys()
        if dict_key not in supported_versions:
            fail(
                ("The supported NCCL versions are {supported_versions}." +
                 " Please provide a supported version in HERMETIC_CUDA_VERSION" +
                 " environment variable or add NCCL distribution for" +
                 " CUDA version={version}, OS={arch}.")
                    .format(
                    supported_versions = supported_versions,
                    version = cuda_version,
                    arch = arch,
                ),
            )
        sha256 = repository_ctx.attr.sha256_dict[dict_key]
        url = repository_ctx.attr.url_dict[dict_key]

        archive_name = get_archive_name(url)
        file_name = archive_name + ".zip"

        print("Downloading and extracting {}".format(url))  # buildifier: disable=print
        repository_ctx.download(
            url = tf_mirror_urls(url),
            output = file_name,
            sha256 = sha256,
        )
        repository_ctx.extract(
            archive = file_name,
            stripPrefix = repository_ctx.attr.strip_prefix,
        )
        repository_ctx.delete(file_name)

        major_version = create_build_file_and_return_redist_major_version(
            repository_ctx,
            major_version_to_build_template_dict,
            cuda_version,
        )
    else:
        # If no CUDA version is found, comment out cc_import targets.
        create_dummy_build_file(repository_ctx, major_version_to_build_template_dict)
    repository_ctx.file("version.txt", major_version)

def _cuda_nccl_repo_impl(repository_ctx):
    local_nccl_path = get_env_var(repository_ctx, "LOCAL_NCCL_PATH")
    if local_nccl_path:
        use_local_cuda_path(repository_ctx, local_nccl_path)
    else:
        _use_downloaded_nccl_wheel(repository_ctx)

_cuda_nccl_repo = repository_rule(
    implementation = _cuda_nccl_repo_impl,
    attrs = {
        "sha256_dict": attr.string_dict(mandatory = True),
        "url_dict": attr.string_dict(mandatory = True),
        "version_dict": attr.string_dict(mandatory = True),
        "build_template_to_major_version_dict": attr.label_keyed_string_dict(mandatory = True),
        "strip_prefix": attr.string(),
    },
    environ = ["HERMETIC_CUDA_VERSION", "TF_CUDA_VERSION", "LOCAL_NCCL_PATH"],
)

def cuda_nccl_repo(
        name,
        sha256_dict,
        url_dict,
        version_dict,
        build_template_to_major_version_dict,
        **kwargs):
    _cuda_nccl_repo(
        name = name,
        sha256_dict = sha256_dict,
        url_dict = url_dict,
        version_dict = version_dict,
        build_template_to_major_version_dict = build_template_to_major_version_dict,
        **kwargs
    )

def hermetic_nccl_redist_init_repository(cuda_nccl_wheels):
    nccl_artifacts_dict = {"sha256_dict": {}, "url_dict": {}, "version_dict": {}}
    for cuda_version, nccl_wheel_info in cuda_nccl_wheels.items():
        for arch in OS_ARCH_DICT.values():
            if arch in nccl_wheel_info.keys():
                cuda_version_to_arch_key = "%s-%s" % (cuda_version, arch)
                nccl_artifacts_dict["sha256_dict"][cuda_version_to_arch_key] = nccl_wheel_info[arch].get("sha256", "")
                nccl_artifacts_dict["url_dict"][cuda_version_to_arch_key] = nccl_wheel_info[arch]["url"]
                nccl_artifacts_dict["version_dict"][cuda_version_to_arch_key] = nccl_wheel_info[arch]["version"]

    cuda_nccl_repo(
        name = "cuda_nccl",
        sha256_dict = nccl_artifacts_dict["sha256_dict"],
        url_dict = nccl_artifacts_dict["url_dict"],
        version_dict = nccl_artifacts_dict["version_dict"],
        build_template_to_major_version_dict = {Label("//third_party/nccl:cuda_nccl.BUILD.tpl"): "default"},
        strip_prefix = "nvidia/nccl",
    )
