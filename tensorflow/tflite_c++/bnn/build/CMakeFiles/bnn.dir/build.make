# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/tf/bnn/tflite_c++/bnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/tf/bnn/tflite_c++/bnn/build

# Include any dependencies generated for this target.
include CMakeFiles/bnn.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/bnn.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/bnn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bnn.dir/flags.make

CMakeFiles/bnn.dir/main.cpp.o: CMakeFiles/bnn.dir/flags.make
CMakeFiles/bnn.dir/main.cpp.o: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/tf/bnn/tflite_c++/bnn/main.cpp
CMakeFiles/bnn.dir/main.cpp.o: CMakeFiles/bnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/tf/bnn/tflite_c++/bnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/bnn.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/bnn.dir/main.cpp.o -MF CMakeFiles/bnn.dir/main.cpp.o.d -o CMakeFiles/bnn.dir/main.cpp.o -c /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/tf/bnn/tflite_c++/bnn/main.cpp

CMakeFiles/bnn.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bnn.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/tf/bnn/tflite_c++/bnn/main.cpp > CMakeFiles/bnn.dir/main.cpp.i

CMakeFiles/bnn.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bnn.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/tf/bnn/tflite_c++/bnn/main.cpp -o CMakeFiles/bnn.dir/main.cpp.s

# Object files for target bnn
bnn_OBJECTS = \
"CMakeFiles/bnn.dir/main.cpp.o"

# External object files for target bnn
bnn_EXTERNAL_OBJECTS =

bnn: CMakeFiles/bnn.dir/main.cpp.o
bnn: CMakeFiles/bnn.dir/build.make
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_stitching.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_superres.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_videostab.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_aruco.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_bgsegm.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_bioinspired.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_ccalib.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_dnn_objdetect.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_dpm.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_face.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_freetype.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_fuzzy.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_hfs.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_img_hash.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_line_descriptor.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_optflow.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_reg.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_rgbd.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_saliency.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_stereo.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_structured_light.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_surface_matching.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_tracking.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_xfeatures2d.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_ximgproc.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_xobjdetect.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_xphoto.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_shape.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_photo.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_phase_unwrapping.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_video.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_datasets.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_plot.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_text.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_dnn.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_ml.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_objdetect.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_calib3d.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_features2d.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_flann.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_highgui.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_videoio.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_imgcodecs.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_imgproc.so.3.4.4
bnn: /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/imgage-processing-workspace/opencv/build/lib/libopencv_core.so.3.4.4
bnn: CMakeFiles/bnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/tf/bnn/tflite_c++/bnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bnn"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bnn.dir/build: bnn
.PHONY : CMakeFiles/bnn.dir/build

CMakeFiles/bnn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bnn.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bnn.dir/clean

CMakeFiles/bnn.dir/depend:
	cd /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/tf/bnn/tflite_c++/bnn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/tf/bnn/tflite_c++/bnn /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/tf/bnn/tflite_c++/bnn /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/tf/bnn/tflite_c++/bnn/build /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/tf/bnn/tflite_c++/bnn/build /media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/quantization/tf/bnn/tflite_c++/bnn/build/CMakeFiles/bnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bnn.dir/depend

