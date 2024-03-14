cmake_minimum_required(VERSION 3.10)
project(ccandle)
set(CCANDLE_USER_CARGO_BUILD_OPTIONS "" CACHE STRING "Additional cargo flags (such as --features) to apply to the build command")
option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
option(CCANDLE_ALWAYS_BUILD "If cmake should always invoke cargo to build ccandle" OFF)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
	set(CCANDLE_BUILD_TYPE_FLAG "--release")
	set(CCANDLE_BUILD_TYPE "release")
else()
	set(CCANDLE_BUILD_TYPE "debug")
endif()

if(BUILD_SHARED_LIBS)
          set(CCANDLE_BUILD_PRODUCT
                  ${CMAKE_CURRENT_SOURCE_DIR}/target/${CCANDLE_BUILD_TYPE}/libccandle.so)
else()
          set(CCANDLE_BUILD_PRODUCT
                  ${CMAKE_CURRENT_SOURCE_DIR}/target/${CCANDLE_BUILD_TYPE}/libccandle.a)
endif()

include(ExternalProject)
find_program(CCANDLE_CARGO_BINARY cargo)
ExternalProject_Add(
	ccandle-crate
	DOWNLOAD_COMMAND ""
	CONFIGURE_COMMAND ""
	INSTALL_COMMAND "${CCANDLE_INSTALL_COMMAND}"
        BUILD_COMMAND ${CCANDLE_PREBUILD_COMMAND} ${CCANDLE_CARGO_BINARY} build ${CCANDLE_BUILD_TYPE_FLAG} ${CCANDLE_USER_CARGO_BUILD_OPTIONS} ${CCANDLE_BUILD_TARGET}
	BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}
        BUILD_ALWAYS ${CCANDLE_ALWAYS_BUILD}
	BUILD_BYPRODUCTS ${CCANDLE_BUILD_PRODUCT})
add_library(ccandle INTERFACE)
add_dependencies(ccandle ccandle-crate)

if(BUILD_SHARED_LIBS)
	if(NOT WIN32)
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath='$ORIGIN'")
	endif()
	target_link_libraries(ccandle INTERFACE ${CCANDLE_BUILD_PRODUCT})
else()
        target_link_libraries(ccandle INTERFACE ${CCANDLE_BUILD_PRODUCT}
                pthread dl m)
endif()

target_include_directories(ccandle INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

include(GNUInstallDirs)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ccandle
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CCANDLE_BUILD_PRODUCT}
	DESTINATION ${CMAKE_INSTALL_LIBDIR})
