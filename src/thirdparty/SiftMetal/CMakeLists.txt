# SiftMetal - Metal-accelerated SIFT feature extraction.
# Compiles Metal shaders to .metallib and builds Obj-C++ host code.

set(SIFT_METAL_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SIFT_METAL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# List all Metal shader sources.
set(METAL_SHADERS
    ${SIFT_METAL_SHADER_DIR}/BilinearUpScale.metal
    ${SIFT_METAL_SHADER_DIR}/Convolution.metal
    ${SIFT_METAL_SHADER_DIR}/ConvolutionSeries.metal
    ${SIFT_METAL_SHADER_DIR}/NearestNeighborDownScale.metal
    ${SIFT_METAL_SHADER_DIR}/SIFTDescriptor.metal
    ${SIFT_METAL_SHADER_DIR}/SIFTExtrema.metal
    ${SIFT_METAL_SHADER_DIR}/SIFTGradient.metal
    ${SIFT_METAL_SHADER_DIR}/SIFTInterpolate.metal
    ${SIFT_METAL_SHADER_DIR}/SIFTOrientation.metal
    ${SIFT_METAL_SHADER_DIR}/Subtract.metal
)

set(METALLIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sift.metallib")

# Compile each .metal to .air, then link into .metallib.
set(AIR_FILES)
foreach(SHADER ${METAL_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(AIR_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.air")
    add_custom_command(
        OUTPUT ${AIR_FILE}
        COMMAND xcrun -sdk macosx metal
            -c ${SHADER}
            -o ${AIR_FILE}
            -I ${SIFT_METAL_INCLUDE_DIR}
            -I ${SIFT_METAL_SHADER_DIR}
        DEPENDS ${SHADER}
        COMMENT "Compiling Metal shader ${SHADER_NAME}.metal"
    )
    list(APPEND AIR_FILES ${AIR_FILE})
endforeach()

add_custom_command(
    OUTPUT ${METALLIB_OUTPUT}
    COMMAND xcrun -sdk macosx metallib ${AIR_FILES} -o ${METALLIB_OUTPUT}
    DEPENDS ${AIR_FILES}
    COMMENT "Linking Metal library sift.metallib"
)

add_custom_target(sift_metallib DEPENDS ${METALLIB_OUTPUT})

# Build the Obj-C++ host library.
add_library(colmap_sift_metal STATIC
    SiftMetal.mm
    SiftMetal.h
)

add_dependencies(colmap_sift_metal sift_metallib)

target_include_directories(colmap_sift_metal PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${SIFT_METAL_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(colmap_sift_metal PRIVATE
    SIFT_METAL_METALLIB_PATH="${METALLIB_OUTPUT}"
)

# Obj-C++ compilation flags.
set_source_files_properties(SiftMetal.mm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc -std=c++17"
)

# Link Metal and Foundation frameworks.
target_link_libraries(colmap_sift_metal PUBLIC
    "-framework Metal"
    "-framework Foundation"
)

# Install metallib alongside binaries.
install(FILES ${METALLIB_OUTPUT}
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
