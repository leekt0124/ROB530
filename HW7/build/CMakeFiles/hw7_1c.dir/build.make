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
CMAKE_SOURCE_DIR = /home/leekt/UMich/ROB530/HW7

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/leekt/UMich/ROB530/HW7/build

# Include any dependencies generated for this target.
include CMakeFiles/hw7_1c.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/hw7_1c.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hw7_1c.dir/flags.make

CMakeFiles/hw7_1c.dir/hw7_1c.cpp.o: CMakeFiles/hw7_1c.dir/flags.make
CMakeFiles/hw7_1c.dir/hw7_1c.cpp.o: ../hw7_1c.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/leekt/UMich/ROB530/HW7/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hw7_1c.dir/hw7_1c.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hw7_1c.dir/hw7_1c.cpp.o -c /home/leekt/UMich/ROB530/HW7/hw7_1c.cpp

CMakeFiles/hw7_1c.dir/hw7_1c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hw7_1c.dir/hw7_1c.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/leekt/UMich/ROB530/HW7/hw7_1c.cpp > CMakeFiles/hw7_1c.dir/hw7_1c.cpp.i

CMakeFiles/hw7_1c.dir/hw7_1c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hw7_1c.dir/hw7_1c.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/leekt/UMich/ROB530/HW7/hw7_1c.cpp -o CMakeFiles/hw7_1c.dir/hw7_1c.cpp.s

# Object files for target hw7_1c
hw7_1c_OBJECTS = \
"CMakeFiles/hw7_1c.dir/hw7_1c.cpp.o"

# External object files for target hw7_1c
hw7_1c_EXTERNAL_OBJECTS =

bin/hw7_1c: CMakeFiles/hw7_1c.dir/hw7_1c.cpp.o
bin/hw7_1c: CMakeFiles/hw7_1c.dir/build.make
bin/hw7_1c: /usr/local/lib/libgtsam.so.4.2.0
bin/hw7_1c: /usr/lib/x86_64-linux-gnu/libboost_serialization.so.1.71.0
bin/hw7_1c: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
bin/hw7_1c: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.71.0
bin/hw7_1c: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.71.0
bin/hw7_1c: /usr/lib/x86_64-linux-gnu/libboost_atomic.so.1.71.0
bin/hw7_1c: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.71.0
bin/hw7_1c: /usr/lib/x86_64-linux-gnu/libboost_regex.so.1.71.0
bin/hw7_1c: /usr/lib/x86_64-linux-gnu/libboost_timer.so.1.71.0
bin/hw7_1c: /usr/lib/x86_64-linux-gnu/libboost_chrono.so.1.71.0
bin/hw7_1c: /usr/local/lib/libmetis-gtsam.so
bin/hw7_1c: CMakeFiles/hw7_1c.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/leekt/UMich/ROB530/HW7/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/hw7_1c"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hw7_1c.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hw7_1c.dir/build: bin/hw7_1c

.PHONY : CMakeFiles/hw7_1c.dir/build

CMakeFiles/hw7_1c.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hw7_1c.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hw7_1c.dir/clean

CMakeFiles/hw7_1c.dir/depend:
	cd /home/leekt/UMich/ROB530/HW7/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/leekt/UMich/ROB530/HW7 /home/leekt/UMich/ROB530/HW7 /home/leekt/UMich/ROB530/HW7/build /home/leekt/UMich/ROB530/HW7/build /home/leekt/UMich/ROB530/HW7/build/CMakeFiles/hw7_1c.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hw7_1c.dir/depend

