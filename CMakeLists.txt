cmake_minimum_required (VERSION 2.8)
set (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

project (kpsf CXX)

# Configure the path structure.
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RELEASE_POSTFIX "")
set (CMAKE_DEBUG_POSTFIX "-debug")

#
# Find the required libraries.
#
# Add Ceres.
find_package (Ceres REQUIRED)
include_directories (${CERES_INCLUDES})

# Add glog and gflags.
find_package (Glog REQUIRED)
include_directories (${GLOG_INCLUDES})
find_package (Gflags REQUIRED)
include_directories (${GFLAGS_INCLUDES})

# Add cfitsio.
find_package (Cfitsio REQUIRED)
include_directories (${CFITSIO_INCLUDES})

#
# Optimizations.
#

if (CMAKE_COMPILER_IS_GNUCXX)
  # Linux
  if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    if (NOT GCC_VERSION VERSION_LESS 4.2)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
    endif (NOT GCC_VERSION VERSION_LESS 4.2)
  endif (CMAKE_SYSTEM_NAME MATCHES "Linux")

  # Mac OS X
  if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
      # Use of -fast only applicable for Apple's GCC
      # Assume this is being used if GCC version < 4.3 on OSX
      execute_process (COMMAND ${CMAKE_C_COMPILER}
                       ARGS ${CMAKE_CXX_COMPILER_ARG1} -dumpversion
                       OUTPUT_VARIABLE GCC_VERSION
                       OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (GCC_VERSION VERSION_LESS 4.3)
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fast")
    endif (GCC_VERSION VERSION_LESS 4.3)
  endif (CMAKE_SYSTEM_NAME MATCHES "Darwin")
endif (CMAKE_COMPILER_IS_GNUCXX)

# Use of -O4 requires use of gold linker & LLVM-gold plugin, which might
# well not be present / in use and without which files will compile, but
# not link ('file not recognized') so explicitly check for support
include (CheckCXXCompilerFlag)
check_cxx_compiler_flag ("-O4" HAVE_LTO_SUPPORT)
if (HAVE_LTO_SUPPORT)
  message (STATUS "Enabling link-time optimization (-O4)")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4")
else ()
  message (STATUS
           "Compiler does not support link-time optimization (-O4).")
endif (HAVE_LTO_SUPPORT)

# Add the solver.
add_subdirectory (src)
