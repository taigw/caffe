# The option allows to include in build only selected test files and exclude all others
# Usage example:
#  cmake -DBUILD_only_tests="common,net,blob,im2col_kernel"
set(BUILD_only_crftests "" CACHE STRING "Blank or comma-separated list of test files to build without ‘crftest_' prefix and extention")
caffe_leave_only_selected_tests(crftest_srcs ${BUILD_only_crftests})
caffe_leave_only_selected_tests(crftest_cuda ${BUILD_only_crftests})

# For 'make runtest' target we don't need to embed test data paths to
# source files, because test target is executed in source directory
# That's why the lines below are commented. TODO: remove them

# definition needed to include CMake generated files
#add_definitions(-DCMAKE_BUILD)

# generates test_data/sample_data_list.txt.gen.cmake
#caffe_configure_testdatafile(test_data/sample_data_list.txt)

set(the_target crftest.testbin)
set(crftest_args --gtest_shuffle)

if(HAVE_CUDA)
  caffe_cuda_compile(crftest_cuda_objs ${crftest_cuda})
  list(APPEND crftest_srcs ${crftest_cuda_objs} ${crftest_cuda})
else()
  list(APPEND crftest_args --gtest_filter="-*GPU*")
endif()

# ---[ Adding test target
add_executable(${the_target} EXCLUDE_FROM_ALL ${crftest_srcs})
target_link_libraries(${the_target} gtest ${Caffe_LINK})
caffe_default_properties(${the_target})
caffe_set_runtime_directory(${the_target} "${PROJECT_BINARY_DIR}/test_crf")

# ---[ Adding runtest
add_custom_target(runcrftest COMMAND ${the_target} ${crftest_args}
                          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
