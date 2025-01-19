cmake_minimum_required(VERSION 3.15)
project(HGS_CVRP)
set(CMAKE_CXX_STANDARD 17)

set(
        src_files
        Program/Genetic.cpp
        Program/Individual.cpp
        Program/LocalSearch.cpp
        Program/Params.cpp
        Program/Population.cpp
        Program/Split.cpp
        Program/InstanceCVRPLIB.cpp
        Program/AlgorithmParameters.cpp
        Program/C_Interface.cpp)

if (MSVC)
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif (MSVC)

include_directories(Program)

# Build Executable
add_executable(bin
        Program/main.cpp
        ${src_files})

set_target_properties(bin PROPERTIES OUTPUT_NAME hgs)

# Build Library
add_library(lib SHARED ${src_files})
set_target_properties(lib PROPERTIES OUTPUT_NAME hgscvrp)


# Install
install(TARGETS lib
        DESTINATION lib)
install(TARGETS bin
        DESTINATION bin)
install(FILES Program/AlgorithmParameters.h Program/C_Interface.h
        DESTINATION include)