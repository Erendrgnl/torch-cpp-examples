# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/eren/Codes/C++/torch_examples/02_linear_regression

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/eren/Codes/C++/torch_examples/02_linear_regression/build

# Include any dependencies generated for this target.
include CMakeFiles/example.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/example.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/example.dir/flags.make

CMakeFiles/example.dir/linear_regression.cpp.o: CMakeFiles/example.dir/flags.make
CMakeFiles/example.dir/linear_regression.cpp.o: ../linear_regression.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eren/Codes/C++/torch_examples/02_linear_regression/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/example.dir/linear_regression.cpp.o"
	/bin/x86_64-linux-gnu-g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example.dir/linear_regression.cpp.o -c /home/eren/Codes/C++/torch_examples/02_linear_regression/linear_regression.cpp

CMakeFiles/example.dir/linear_regression.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example.dir/linear_regression.cpp.i"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eren/Codes/C++/torch_examples/02_linear_regression/linear_regression.cpp > CMakeFiles/example.dir/linear_regression.cpp.i

CMakeFiles/example.dir/linear_regression.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example.dir/linear_regression.cpp.s"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eren/Codes/C++/torch_examples/02_linear_regression/linear_regression.cpp -o CMakeFiles/example.dir/linear_regression.cpp.s

# Object files for target example
example_OBJECTS = \
"CMakeFiles/example.dir/linear_regression.cpp.o"

# External object files for target example
example_EXTERNAL_OBJECTS =

example: CMakeFiles/example.dir/linear_regression.cpp.o
example: CMakeFiles/example.dir/build.make
example: /home/eren/anaconda3/envs/teknofest_yarisma/lib/python3.7/site-packages/torch/lib/libtorch.so
example: /home/eren/anaconda3/envs/teknofest_yarisma/lib/python3.7/site-packages/torch/lib/libc10.so
example: /usr/local/cuda-11.1/lib64/stubs/libcuda.so
example: /usr/local/cuda-11.1/lib64/libnvrtc.so
example: /usr/local/cuda-11.1/lib64/libnvToolsExt.so
example: /usr/local/cuda-11.1/lib64/libcudart.so
example: /home/eren/anaconda3/envs/teknofest_yarisma/lib/python3.7/site-packages/torch/lib/libc10_cuda.so
example: /home/eren/anaconda3/envs/teknofest_yarisma/lib/python3.7/site-packages/torch/lib/libc10_cuda.so
example: /home/eren/anaconda3/envs/teknofest_yarisma/lib/python3.7/site-packages/torch/lib/libc10.so
example: /usr/local/cuda-11.1/lib64/libcufft.so
example: /usr/local/cuda-11.1/lib64/libcurand.so
example: /usr/local/cuda-11.1/lib64/libcublas.so
example: /usr/local/cuda-11.1/lib64/libcudnn.so
example: /usr/local/cuda-11.1/lib64/libnvToolsExt.so
example: /usr/local/cuda-11.1/lib64/libcudart.so
example: CMakeFiles/example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/eren/Codes/C++/torch_examples/02_linear_regression/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/example.dir/build: example

.PHONY : CMakeFiles/example.dir/build

CMakeFiles/example.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/example.dir/cmake_clean.cmake
.PHONY : CMakeFiles/example.dir/clean

CMakeFiles/example.dir/depend:
	cd /home/eren/Codes/C++/torch_examples/02_linear_regression/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/eren/Codes/C++/torch_examples/02_linear_regression /home/eren/Codes/C++/torch_examples/02_linear_regression /home/eren/Codes/C++/torch_examples/02_linear_regression/build /home/eren/Codes/C++/torch_examples/02_linear_regression/build /home/eren/Codes/C++/torch_examples/02_linear_regression/build/CMakeFiles/example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/example.dir/depend

