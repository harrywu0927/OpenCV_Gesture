# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

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
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.22.2/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.22.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/harrywu/Documents/gesture

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/harrywu/Documents/gesture

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/opt/homebrew/Cellar/cmake/3.22.2/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/opt/homebrew/Cellar/cmake/3.22.2/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/harrywu/Documents/gesture/CMakeFiles /Users/harrywu/Documents/gesture//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/harrywu/Documents/gesture/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named main

# Build rule for target.
main: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 main
.PHONY : main

# fast build rule for target.
main/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/build
.PHONY : main/fast

#=============================================================================
# Target rules for targets named objdetect

# Build rule for target.
objdetect: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 objdetect
.PHONY : objdetect

# fast build rule for target.
objdetect/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/objdetect.dir/build.make CMakeFiles/objdetect.dir/build
.PHONY : objdetect/fast

#=============================================================================
# Target rules for targets named writeFeatures

# Build rule for target.
writeFeatures: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 writeFeatures
.PHONY : writeFeatures

# fast build rule for target.
writeFeatures/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/writeFeatures.dir/build.make CMakeFiles/writeFeatures.dir/build
.PHONY : writeFeatures/fast

# target to build an object file
main.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/main.o
.PHONY : main.o

# target to preprocess a source file
main.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/main.i
.PHONY : main.i

# target to generate assembly for a file
main.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/main.s
.PHONY : main.s

# target to build an object file
objectDetection.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/objdetect.dir/build.make CMakeFiles/objdetect.dir/objectDetection.o
.PHONY : objectDetection.o

# target to preprocess a source file
objectDetection.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/objdetect.dir/build.make CMakeFiles/objdetect.dir/objectDetection.i
.PHONY : objectDetection.i

# target to generate assembly for a file
objectDetection.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/objdetect.dir/build.make CMakeFiles/objdetect.dir/objectDetection.s
.PHONY : objectDetection.s

# target to build an object file
writeFeatures.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/writeFeatures.dir/build.make CMakeFiles/writeFeatures.dir/writeFeatures.o
.PHONY : writeFeatures.o

# target to preprocess a source file
writeFeatures.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/writeFeatures.dir/build.make CMakeFiles/writeFeatures.dir/writeFeatures.i
.PHONY : writeFeatures.i

# target to generate assembly for a file
writeFeatures.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/writeFeatures.dir/build.make CMakeFiles/writeFeatures.dir/writeFeatures.s
.PHONY : writeFeatures.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... main"
	@echo "... objdetect"
	@echo "... writeFeatures"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
	@echo "... objectDetection.o"
	@echo "... objectDetection.i"
	@echo "... objectDetection.s"
	@echo "... writeFeatures.o"
	@echo "... writeFeatures.i"
	@echo "... writeFeatures.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

