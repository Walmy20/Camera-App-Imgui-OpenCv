cmake_minimum_required(VERSION 3.0)
project(CameraApp)

# Set output directory
set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/build)

# Source files
file(GLOB SRC "*.cpp" "imgui/*.cpp")

# Include directories
include_directories(include include/imgui)

# Link directories
link_directories(libs)

# Find packages
find_package(OpenGL REQUIRED)

# Executable
add_executable(${PROJECT_NAME} ${SRC})

# Set output directory for executable
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE glfw ${OPENGL_gl_LIBRARY})

# Optionally link other libraries if needed
# target_link_libraries(${PROJECT_NAME} PRIVATE other_library)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Werror")

# Copy resources or other files if needed
# file(COPY ${CMAKE_SOURCE_DIR}/resources DESTINATION ${OUTPUT_DIR})

# Optionally set additional compiler or linker flags
# target_compile_options(${PROJECT_NAME} PRIVATE -O3)

