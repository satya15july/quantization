load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "label_image",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "label_image.cc",
     ],
    linkopts = tflite_linkopts() + select({
        "@org_tensorflow//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//larq_compute_engine/tflite/kernels:lce_op_kernels",
        #"@opencv//:opencv",
        #":bitmap_helpers",
        #"@org_tensorflow//tensorflow/lite:framework",
        #"@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        #"@com_google_absl//absl/memory",
        #"@com_google_absl//absl/strings",

        ":bitmap_helpers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/profiling:profiler",
        "@org_tensorflow//tensorflow/lite/tools:command_line_flags",
        "@org_tensorflow//tensorflow/lite/tools:tool_params",
        "@org_tensorflow//tensorflow/lite/tools/delegates:delegate_provider_hdr",
        "@org_tensorflow//tensorflow/lite/tools/delegates:tflite_execution_providers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "bitmap_helpers",
    srcs = ["bitmap_helpers.cc"],
    hdrs = [
        "bitmap_helpers.h",
        "bitmap_helpers_impl.h",
        "label_image.h",
        "log.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            "@org_tensorflow//tensorflow/lite/delegates/gpu:delegate",
            "//tensorflow/lite/delegates/hexagon:hexagon_delegate",
        ],
        "@org_tensorflow//tensorflow:android_arm64": [
            "@org_tensorflow//tensorflow/lite/delegates/gpu:delegate",
            "@org_tensorflow//tensorflow/lite/delegates/hexagon:hexagon_delegate",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "label_image_test",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "label_image_test.cc",
    ],
    data = [
        "testdata/grace_hopper.bmp",
    ],
    deps = [
        ":bitmap_helpers",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)