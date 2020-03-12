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

# Utility rule file for Betl2.

# Include the progress variables for this target.
include external/Code/CMakeFiles/Betl2.dir/progress.make

external/Code/CMakeFiles/Betl2: external/Code/CMakeFiles/Betl2-complete


external/Code/CMakeFiles/Betl2-complete: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-install
external/Code/CMakeFiles/Betl2-complete: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-mkdir
external/Code/CMakeFiles/Betl2-complete: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-download
external/Code/CMakeFiles/Betl2-complete: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-update
external/Code/CMakeFiles/Betl2-complete: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-patch
external/Code/CMakeFiles/Betl2-complete: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-configure
external/Code/CMakeFiles/Betl2-complete: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-build
external/Code/CMakeFiles/Betl2-complete: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/u/ppanchal/ETH/fcsc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'Betl2'"
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E make_directory /u/ppanchal/ETH/fcsc/build/external/Code/CMakeFiles
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E touch /u/ppanchal/ETH/fcsc/build/external/Code/CMakeFiles/Betl2-complete
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E touch /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-done

external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-install: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/u/ppanchal/ETH/fcsc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Performing install step for 'Betl2'"
	cd /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-build && $(MAKE) install
	cd /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-build && /usr/bin/cmake -E touch /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-install

external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/u/ppanchal/ETH/fcsc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Creating directories for 'Betl2'"
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E make_directory /u/ppanchal/ETH/fcsc/external/Code/third_party/Betl2/Library
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E make_directory /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-build
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E make_directory /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E make_directory /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/tmp
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E make_directory /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-stamp
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E make_directory /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E make_directory /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-stamp
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E touch /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-mkdir

external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-download: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/u/ppanchal/ETH/fcsc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "No download step for 'Betl2'"
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E echo_append
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E touch /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-download

external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-update: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/u/ppanchal/ETH/fcsc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "No update step for 'Betl2'"
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E echo_append
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E touch /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-update

external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-patch: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/u/ppanchal/ETH/fcsc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "No patch step for 'Betl2'"
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E echo_append
	cd /u/ppanchal/ETH/fcsc/build/external/Code && /usr/bin/cmake -E touch /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-patch

external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-configure: external/Code/Betl2-prefix/tmp/Betl2-cfgcmd.txt
external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-configure: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-update
external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-configure: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/u/ppanchal/ETH/fcsc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Performing configure step for 'Betl2'"
	cd /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-build && /usr/bin/cmake -DCMAKE_INSTALL_PREFIX=/u/ppanchal/ETH/fcsc/build/external/Code/betl2_install -DETH_INCLUDE_DIRS=/u/ppanchal/ETH/fcsc/build/external/Code/ethGenericGrid_install/include/eth /u/ppanchal/ETH/fcsc/build/Eigen_install/include -DETH_BASE_LIB=/u/ppanchal/ETH/fcsc/build/external/Code/ethGenericGrid_install/lib -DCMAKE_INSTALL_PREFIX:PATH=/u/ppanchal/ETH/fcsc/build/external/Code/betl2_install -DCMAKE_PREFIX_PATH=/u/ppanchal/ETH/fcsc/build/Eigen_install -DETH_ROOT=/u/ppanchal/ETH/fcsc/build/external/Code/ethGenericGrid_install -DCMAKE_INCLUDE_DIRECTORIES_BEFORE= "-GUnix Makefiles" /u/ppanchal/ETH/fcsc/external/Code/third_party/Betl2/Library
	cd /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-build && /usr/bin/cmake -E touch /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-configure

external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-build: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/u/ppanchal/ETH/fcsc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Performing build step for 'Betl2'"
	cd /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-build && $(MAKE)
	cd /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-build && /usr/bin/cmake -E touch /u/ppanchal/ETH/fcsc/build/external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-build

Betl2: external/Code/CMakeFiles/Betl2
Betl2: external/Code/CMakeFiles/Betl2-complete
Betl2: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-install
Betl2: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-mkdir
Betl2: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-download
Betl2: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-update
Betl2: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-patch
Betl2: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-configure
Betl2: external/Code/Betl2-prefix/src/Betl2-stamp/Betl2-build
Betl2: external/Code/CMakeFiles/Betl2.dir/build.make

.PHONY : Betl2

# Rule to build all files generated by this target.
external/Code/CMakeFiles/Betl2.dir/build: Betl2

.PHONY : external/Code/CMakeFiles/Betl2.dir/build

external/Code/CMakeFiles/Betl2.dir/clean:
	cd /u/ppanchal/ETH/fcsc/build/external/Code && $(CMAKE_COMMAND) -P CMakeFiles/Betl2.dir/cmake_clean.cmake
.PHONY : external/Code/CMakeFiles/Betl2.dir/clean

external/Code/CMakeFiles/Betl2.dir/depend:
	cd /u/ppanchal/ETH/fcsc/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /u/ppanchal/ETH/fcsc /u/ppanchal/ETH/fcsc/external/Code /u/ppanchal/ETH/fcsc/build /u/ppanchal/ETH/fcsc/build/external/Code /u/ppanchal/ETH/fcsc/build/external/Code/CMakeFiles/Betl2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : external/Code/CMakeFiles/Betl2.dir/depend

