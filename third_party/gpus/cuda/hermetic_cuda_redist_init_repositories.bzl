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

"""Hermetic CUDA repositories initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_mirror_urls")

OS_ARCH_DICT = {
    "amd64": "x86_64-unknown-linux-gnu",
    "aarch64": "aarch64-unknown-linux-gnu",
}
_REDIST_ARCH_DICT = {
    "linux-x86_64": "x86_64-unknown-linux-gnu",
    "linux-sbsa": "aarch64-unknown-linux-gnu",
}

SUPPORTED_ARCHIVE_EXTENSIONS = [
    ".zip",
    ".jar",
    ".war",
    ".aar",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.xz",
    ".txz",
    ".tar.zst",
    ".tzst",
    ".tar.bz2",
    ".tbz",
    ".ar",
    ".deb",
    ".whl",
]

DIST_NAME_TO_CUDA_REPOSITORY = {
    "libcublas": "cuda_cublas",
    "cuda_cudart": "cuda_cudart",
    "libcufft": "cuda_cufft",
    "cuda_cupti": "cuda_cupti",
    "libcurand": "cuda_curand",
    "libcusolver": "cuda_cusolver",
    "libcusparse": "cuda_cusparse",
    "libnvjitlink": "cuda_nvjitlink",
    "cuda_nvrtc": "cuda_nvrtc",
    "cuda_cccl": "cuda_cccl",
    "cuda_nvcc": "cuda_nvcc",
    "cuda_nvml_dev": "cuda_nvml",
    "cuda_nvprune": "cuda_nvprune",
    "cuda_nvtx": "cuda_nvtx",
}

def get_env_var(ctx, name):
    if name in ctx.os.environ:
        return ctx.os.environ[name]
    else:
        return None

def _get_file_name(url):
    last_slash_index = url.rfind("/")
    return url[last_slash_index + 1:]

def get_archive_name(url):
    filename = _get_file_name(url)
    for extension in SUPPORTED_ARCHIVE_EXTENSIONS:
        if filename.endswith(extension):
            return filename[:-len(extension)]
    return filename

LIB_EXTENSION = ".so."

def _get_lib_name_and_version(path):
    extension_index = path.rfind(LIB_EXTENSION)
    last_slash_index = path.rfind("/")
    lib_name = path[last_slash_index + 1:extension_index]
    lib_version = path[extension_index + len(LIB_EXTENSION):]
    return (lib_name, lib_version)

def get_lib_name_to_version_dict(repository_ctx):
    lib_name_to_version_dict = {}
    main_lib_name = "lib{}".format(repository_ctx.name.split("_")[1]).lower()
    lib_dir_content = repository_ctx.path("lib").readdir()
    for path in [
        str(f)
        for f in lib_dir_content
        if (LIB_EXTENSION in str(f) and
            main_lib_name in str(f).lower())
    ]:
        lib_name, lib_version = _get_lib_name_and_version(path)
        key = "%%{%s_version}" % lib_name.lower()
        if len(lib_version.split(".")) == 1:
            lib_name_to_version_dict[key] = lib_version
        if (len(lib_version.split(".")) == 2 and
            key not in lib_name_to_version_dict):
            lib_name_to_version_dict[key] = lib_version
    return lib_name_to_version_dict

def create_dummy_build_file(
        repository_ctx,
        major_version_to_build_template_dict):
    repository_ctx.template(
        "BUILD",
        major_version_to_build_template_dict["default"],
        {
            "%{multiline_comment}": "'''",
            "%{comment}": "#",
        },
    )

def create_build_file_and_return_redist_major_version(
        repository_ctx,
        major_version_to_build_template_dict,
        cuda_version = None):
    major_version = ""
    lib_name_to_version_dict = get_lib_name_to_version_dict(repository_ctx)
    if len(lib_name_to_version_dict) == 0:
        create_dummy_build_file(
            repository_ctx,
            major_version_to_build_template_dict,
        )
    else:
        main_lib_name = "lib{}".format(repository_ctx.name.split("_")[1])
        key = "%%{%s_version}" % main_lib_name
        major_version = lib_name_to_version_dict[key]
        repository_ctx.template(
            "BUILD",
            (major_version_to_build_template_dict.get(major_version) or
             major_version_to_build_template_dict["default"]),
            lib_name_to_version_dict | {
                "%{multiline_comment}": "",
                "%{comment}": "",
            },
        )
    return major_version

def _create_symlinks(repository_ctx, local_path, dirs):
    for dir in dirs:
        repository_ctx.symlink(
            "{path}/{dir}".format(
                path = local_path,
                dir = dir,
            ),
            dir,
        )

def use_local_cuda_path(repository_ctx, local_path):
    """ Creates symlinks to dirs inside local CUDA and inits hermetic CUDA
    repository.
    """
    cuda_version = (get_env_var(repository_ctx, "HERMETIC_CUDA_VERSION") or
                    get_env_var(repository_ctx, "TF_CUDA_VERSION"))
    _create_symlinks(
        repository_ctx,
        local_path,
        ["include", "lib", "bin", "nvvm"],
    )
    major_version_to_build_template_dict = {
        v: k
        for k, v in repository_ctx.attr.build_template_to_major_version_dict.items()
    }
    major_version = create_build_file_and_return_redist_major_version(
        repository_ctx,
        major_version_to_build_template_dict,
        cuda_version,
    )
    repository_ctx.file("version.txt", major_version)

def _use_local_cudnn_path(repository_ctx, local_path):
    """ Creates symlinks to dirs inside local CUDNN and inits hermetic CUDNN
    repository.
    """
    _create_symlinks(
        repository_ctx,
        local_path,
        ["include", "lib"],
    )
    major_version_to_build_template_dict = {
        v: k
        for k, v in repository_ctx.attr.build_template_to_major_version_dict.items()
    }
    major_version = create_build_file_and_return_redist_major_version(
        repository_ctx,
        major_version_to_build_template_dict,
    )
    repository_ctx.file("version.txt", major_version)

def _download_redistribution(repository_ctx, arch_key, path_prefix):
    (url, sha256) = repository_ctx.attr.url_dict[arch_key]

    # If url is not relative, then appending prefix is not needed.
    if not (url.startswith("http") or url.startswith("file:///")):
        url = path_prefix + url
    archive_name = get_archive_name(url)
    file_name = _get_file_name(url)

    print("Downloading and extracting {}".format(url))  # buildifier: disable=print
    repository_ctx.download(
        url = tf_mirror_urls(url),
        output = file_name,
        sha256 = sha256,
    )
    if repository_ctx.attr.override_strip_prefix:
        strip_prefix = repository_ctx.attr.override_strip_prefix
    else:
        strip_prefix = archive_name
    repository_ctx.extract(
        archive = file_name,
        stripPrefix = strip_prefix,
    )
    repository_ctx.delete(file_name)

def _use_downloaded_cuda_redistribution(repository_ctx):
    """ Downloads CUDA redistribution and inits hermetic CUDA repository."""
    major_version = ""
    cuda_version = (get_env_var(repository_ctx, "HERMETIC_CUDA_VERSION") or
                    get_env_var(repository_ctx, "TF_CUDA_VERSION"))
    major_version_to_build_template_dict = {
        v: k
        for k, v in repository_ctx.attr.build_template_to_major_version_dict.items()
    }
    if cuda_version:
        # Download archive only when GPU config is used.
        arch_key = OS_ARCH_DICT[repository_ctx.os.arch]
        if arch_key not in repository_ctx.attr.url_dict.keys():
            fail(
                ("The supported platforms are {supported_platforms}." +
                 " Platform {platform} is not supported for {dist_name}.")
                    .format(
                    supported_platforms = repository_ctx.attr.url_dict.keys(),
                    platform = arch_key,
                    dist_name = repository_ctx.name,
                ),
            )
        _download_redistribution(
            repository_ctx,
            arch_key,
            repository_ctx.attr.cuda_dist_path_prefix,
        )

        major_version = create_build_file_and_return_redist_major_version(
            repository_ctx,
            major_version_to_build_template_dict,
            cuda_version,
        )
    else:
        # If no CUDA version is found, comment out all cc_import targets.
        create_dummy_build_file(
            repository_ctx,
            major_version_to_build_template_dict,
        )
    repository_ctx.file("version.txt", major_version)

def _cuda_repo_impl(repository_ctx):
    local_cuda_path = get_env_var(repository_ctx, "LOCAL_CUDA_PATH")
    if local_cuda_path:
        use_local_cuda_path(repository_ctx, local_cuda_path)
    else:
        _use_downloaded_cuda_redistribution(repository_ctx)

_cuda_repo = repository_rule(
    implementation = _cuda_repo_impl,
    attrs = {
        "url_dict": attr.string_list_dict(mandatory = True),
        "build_template_to_major_version_dict": attr.label_keyed_string_dict(mandatory = True),
        "override_strip_prefix": attr.string(),
        "cuda_dist_path_prefix": attr.string(),
    },
    environ = [
        "HERMETIC_CUDA_VERSION",
        "TF_CUDA_VERSION",
        "LOCAL_CUDA_PATH",
    ],
)

def cuda_repo(name, url_dict, build_template_to_major_version_dict, **kwargs):
    _cuda_repo(
        name = name,
        url_dict = url_dict,
        build_template_to_major_version_dict = build_template_to_major_version_dict,
        **kwargs
    )

def _use_downloaded_cudnn_redistribution(repository_ctx):
    """ Downloads CUDNN redistribution and inits hermetic CUDNN repository."""
    cudnn_version = None
    major_version = ""
    cudnn_version = (get_env_var(repository_ctx, "HERMETIC_CUDNN_VERSION") or
                     get_env_var(repository_ctx, "TF_CUDNN_VERSION"))
    cuda_version = (get_env_var(repository_ctx, "HERMETIC_CUDA_VERSION") or
                    get_env_var(repository_ctx, "TF_CUDA_VERSION"))
    major_version_to_build_template_dict = {
        v: k
        for k, v in repository_ctx.attr.build_template_to_major_version_dict.items()
    }
    if cudnn_version:
        # Download archive only when GPU config is used.
        arch_key = OS_ARCH_DICT[repository_ctx.os.arch]
        if arch_key not in repository_ctx.attr.url_dict.keys():
            arch_key = "cuda{version}_{arch}".format(
                version = cuda_version.split(".")[0],
                arch = arch_key,
            )
        if arch_key in repository_ctx.attr.url_dict.keys():
            _download_redistribution(
                repository_ctx,
                arch_key,
                repository_ctx.attr.cudnn_dist_path_prefix,
            )

            major_version = create_build_file_and_return_redist_major_version(
                repository_ctx,
                major_version_to_build_template_dict,
            )
        else:
            fail(
                ("The supported platforms are {supported_platforms}." +
                 " Platform {platform} is not supported for {dist_name}.")
                    .format(
                    supported_platforms = repository_ctx.attr.url_dict.keys(),
                    platform = arch_key,
                    dist_name = repository_ctx.name,
                ),
            )
    else:
        # If no CUDNN version is found, comment out cc_import targets.
        create_dummy_build_file(
            repository_ctx,
            major_version_to_build_template_dict,
        )
    repository_ctx.file("version.txt", major_version)

def _cudnn_repo_impl(repository_ctx):
    local_cudnn_path = get_env_var(repository_ctx, "LOCAL_CUDNN_PATH")
    if local_cudnn_path:
        _use_local_cudnn_path(repository_ctx, local_cudnn_path)
    else:
        _use_downloaded_cudnn_redistribution(repository_ctx)

_cudnn_repo = repository_rule(
    implementation = _cudnn_repo_impl,
    attrs = {
        "url_dict": attr.string_list_dict(mandatory = True),
        "build_template_to_major_version_dict": attr.label_keyed_string_dict(mandatory = True),
        "override_strip_prefix": attr.string(),
        "cudnn_dist_path_prefix": attr.string(),
    },
    environ = [
        "HERMETIC_CUDNN_VERSION",
        "TF_CUDNN_VERSION",
        "HERMETIC_CUDA_VERSION",
        "TF_CUDA_VERSION",
        "LOCAL_CUDNN_PATH",
    ],
)

def cudnn_repo(name, url_dict, build_template_to_major_version_dict, **kwargs):
    _cudnn_repo(
        name = name,
        url_dict = url_dict,
        build_template_to_major_version_dict = build_template_to_major_version_dict,
        **kwargs
    )

def _get_distribution_urls(dist_info):
    url_dict = {}
    for arch in _REDIST_ARCH_DICT.keys():
        if "relative_path" not in dist_info[arch]:
            if "full_path" not in dist_info[arch]:
                for cuda_version, data in dist_info[arch].items():
                    # CUDNN JSON might contain paths for each CUDA version.
                    path_key = "relative_path"
                    if path_key not in data.keys():
                        path_key = "full_path"
                    url_dict["{cuda_version}_{arch}" \
                        .format(
                        cuda_version = cuda_version,
                        arch = _REDIST_ARCH_DICT[arch],
                    )] = [data[path_key], data.get("sha256", "")]
            else:
                url_dict[_REDIST_ARCH_DICT[arch]] = [
                    dist_info[arch]["full_path"],
                    dist_info[arch].get("sha256", ""),
                ]
        else:
            url_dict[_REDIST_ARCH_DICT[arch]] = [
                dist_info[arch]["relative_path"],
                dist_info[arch].get("sha256", ""),
            ]
    return url_dict

def hermetic_cudnn_redist_init_repository(cudnn_dist_path_prefix, cudnn_distributions):
    if "cudnn" in cudnn_distributions.keys():
        url_dict = _get_distribution_urls(cudnn_distributions["cudnn"])
    else:
        url_dict = {"": []}
    cudnn_repo(
        name = "cuda_cudnn",
        build_template_to_major_version_dict = {
            Label("//third_party/gpus/cuda:cuda_cudnn9.BUILD.tpl"): "9",
            Label("//third_party/gpus/cuda:cuda_cudnn.BUILD.tpl"): "default",
        },
        url_dict = url_dict,
        cudnn_dist_path_prefix = cudnn_dist_path_prefix,
    )

def hermetic_cuda_redist_init_repositories(
        cuda_distributions,
        cuda_dist_path_prefix):
    for dist_name, repo_name in DIST_NAME_TO_CUDA_REPOSITORY.items():
        if dist_name in cuda_distributions.keys():
            url_dict = _get_distribution_urls(cuda_distributions[dist_name])
        else:
            url_dict = {"": []}
        cuda_repo(
            name = repo_name,
            build_template_to_major_version_dict = {Label("//third_party/gpus/cuda:{}.BUILD.tpl".format(repo_name)): "default"},
            url_dict = url_dict,
            cuda_dist_path_prefix = cuda_dist_path_prefix,
        )
