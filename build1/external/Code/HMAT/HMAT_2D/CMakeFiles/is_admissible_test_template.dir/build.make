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
include external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/depend.make

# Include the progress variables for this target.
include external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/progress.make

# Include the compile flags for this target's objects.
include external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/flags.make

external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/is_admissible_test.cpp.o: external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/flags.make
external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/is_admissible_test.cpp.o: ../external/Code/HMAT/HMAT_2D/is_admissible_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/u/ppanchal/ETH/fcsc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/is_admissible_test.cpp.o"
	cd /u/ppanchal/ETH/fcsc/build/external/Code/HMAT/HMAT_2D && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/is_admissible_test_template.dir/is_admissible_test.cpp.o -c /u/ppanchal/ETH/fcsc/external/Code/HMAT/HMAT_2D/is_admissible_test.cpp

external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/is_admissible_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/is_admissible_test_template.dir/is_admissible_test.cpp.i"
	cd /u/ppanchal/ETH/fcsc/build/external/Code/HMAT/HMAT_2D && /usr/lib64/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /u/ppanchal/ETH/fcsc/external/Code/HMAT/HMAT_2D/is_admissible_test.cpp > CMakeFiles/is_admissible_test_template.dir/is_admissible_test.cpp.i

external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/is_admissible_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/is_admissible_test_template.dir/is_admissible_test.cpp.s"
	cd /u/ppanchal/ETH/fcsc/build/external/Code/HMAT/HMAT_2D && /usr/lib64/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /u/ppanchal/ETH/fcsc/external/Code/HMAT/HMAT_2D/is_admissible_test.cpp -o CMakeFiles/is_admissible_test_template.dir/is_admissible_test.cpp.s

# Object files for target is_admissible_test_template
is_admissible_test_template_OBJECTS = \
"CMakeFiles/is_admissible_test_template.dir/is_admissible_test.cpp.o"

# External object files for target is_admissible_test_template
is_admissible_test_template_EXTERNAL_OBJECTS =

external/Code/bin/is_admissible_test_template: external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/is_admissible_test.cpp.o
external/Code/bin/is_admissible_test_template: external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/build.make
external/Code/bin/is_admissible_test_template: external/Code/HMAT/HMAT_2D/libhmat_2d_tmps.so
external/Code/bin/is_admissible_test_template: external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/u/ppanchal/ETH/fcsc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/is_admissible_test_template"
	cd /u/ppanchal/ETH/fcsc/build/external/Code/HMAT/HMAT_2D && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/is_admissible_test_template.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/build: external/Code/bin/is_admissible_test_template

.PHONY : external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/build

external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/clean:
	cd /u/ppanchal/ETH/fcsc/build/external/Code/HMAT/HMAT_2D && $(CMAKE_COMMAND) -P CMakeFiles/is_admissible_test_template.dir/cmake_clean.cmake
.PHONY : external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/clean

external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/depend:
	cd /u/ppanchal/ETH/fcsc/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /u/ppanchal/ETH/fcsc /u/ppanchal/ETH/fcsc/external/Code/HMAT/HMAT_2D /u/ppanchal/ETH/fcsc/build /u/ppanchal/ETH/fcsc/build/external/Code/HMAT/HMAT_2D /u/ppanchal/ETH/fcsc/build/external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : external/Code/HMAT/HMAT_2D/CMakeFiles/is_admissible_test_template.dir/depend

