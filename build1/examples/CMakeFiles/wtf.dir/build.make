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
CMAKE_SOURCE_DIR = /u/ppanchal/ETH/fcsc

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /u/ppanchal/ETH/fcsc/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/wtf.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/wtf.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/wtf.dir/flags.make

examples/CMakeFiles/wtf.dir/wtf.cpp.o: examples/CMakeFiles/wtf.dir/flags.make
examples/CMakeFiles/wtf.dir/wtf.cpp.o: ../examples/wtf.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/u/ppanchal/ETH/fcsc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/wtf.dir/wtf.cpp.o"
	cd /u/ppanchal/ETH/fcsc/build/examples && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/wtf.dir/wtf.cpp.o -c /u/ppanchal/ETH/fcsc/examples/wtf.cpp

examples/CMakeFiles/wtf.dir/wtf.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wtf.dir/wtf.cpp.i"
	cd /u/ppanchal/ETH/fcsc/build/examples && /usr/lib64/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /u/ppanchal/ETH/fcsc/examples/wtf.cpp > CMakeFiles/wtf.dir/wtf.cpp.i

examples/CMakeFiles/wtf.dir/wtf.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wtf.dir/wtf.cpp.s"
	cd /u/ppanchal/ETH/fcsc/build/examples && /usr/lib64/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /u/ppanchal/ETH/fcsc/examples/wtf.cpp -o CMakeFiles/wtf.dir/wtf.cpp.s

# Object files for target wtf
wtf_OBJECTS = \
"CMakeFiles/wtf.dir/wtf.cpp.o"

# External object files for target wtf
wtf_EXTERNAL_OBJECTS =

examples/wtf: examples/CMakeFiles/wtf.dir/wtf.cpp.o
examples/wtf: examples/CMakeFiles/wtf.dir/build.make
examples/wtf: external/Code/BEM/2DParametricBEM/src/Parametrizations/libparametrizations.a
examples/wtf: external/Code/BEM/2DParametricBEM/src/libadj_double_layer.a
examples/wtf: external/Code/BEM/2DParametricBEM/src/libhypersingular.a
examples/wtf: external/Code/BEM/2DParametricBEM/src/libsingle_layer.a
examples/wtf: external/Code/BEM/2DParametricBEM/src/libdouble_layer.a
examples/wtf: external/Code/BEM/2DParametricBEM/src/Quadrature/libquadrature.a
examples/wtf: external/Code/BEM/CppHilbert/Library/libCppHilbert.so
examples/wtf: examples/CMakeFiles/wtf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/u/ppanchal/ETH/fcsc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable wtf"
	cd /u/ppanchal/ETH/fcsc/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/wtf.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/wtf.dir/build: examples/wtf

.PHONY : examples/CMakeFiles/wtf.dir/build

examples/CMakeFiles/wtf.dir/clean:
	cd /u/ppanchal/ETH/fcsc/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/wtf.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/wtf.dir/clean

examples/CMakeFiles/wtf.dir/depend:
	cd /u/ppanchal/ETH/fcsc/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /u/ppanchal/ETH/fcsc /u/ppanchal/ETH/fcsc/examples /u/ppanchal/ETH/fcsc/build /u/ppanchal/ETH/fcsc/build/examples /u/ppanchal/ETH/fcsc/build/examples/CMakeFiles/wtf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/wtf.dir/depend

