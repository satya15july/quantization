# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# compile CXX with /usr/bin/c++
CXX_DEFINES = -DUSE_C10D_GLOO -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE

CXX_INCLUDES = -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/pt/bnn/c++_port/libtorch/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/pt/bnn/c++_port/libtorch/include/torch/csrc/api/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/include/opencv -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/core/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/flann/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/imgproc/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/ml/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/phase_unwrapping/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/photo/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/plot/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/reg/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/surface_matching/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/video/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/xphoto/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/dnn/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/freetype/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/fuzzy/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/hfs/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/img_hash/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/imgcodecs/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/shape/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/videoio/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/highgui/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/superres/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/ts/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/bioinspired/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/dnn_objdetect/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/features2d/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/line_descriptor/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/saliency/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/text/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/calib3d/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/ccalib/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/datasets/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/objdetect/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/rgbd/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/stereo/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/structured_light/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/tracking/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/videostab/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/xfeatures2d/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/ximgproc/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/xobjdetect/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/aruco/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/bgsegm/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/dpm/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/face/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv_contrib/modules/optflow/include -isystem /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/modules/stitching/include

CXX_FLAGS = -D_GLIBCXX_USE_CXX11_ABI=1 -std=gnu++14
