cmake_minimum_required(VERSION 3.12)
project(NoCDAS LANGUAGES CXX)

option(CNOC_MODE "Build the simulator with cNoC in-transit execution enabled" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/NoC/*.cpp")
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "src/*.hpp" "src/NoC/*.hpp")

add_executable(NoCDASim ${SOURCES} ${HEADERS})

target_include_directories(NoCDASim PRIVATE src src/NoC src/input)

if(CNOC_MODE)
    target_compile_definitions(NoCDASim PRIVATE cNoC_MODE)
    set(NOCDAS_RUN_MODE cnoc)
else()
    set(NOCDAS_RUN_MODE baseline)
endif()

# Parameterized workload runner.  MODEL and EXP are expanded by the generated
# Makefile, e.g. `make run_workload MODEL=phi EXP=1`.
add_custom_target(run_workload
    COMMAND ${CMAKE_COMMAND}
        -DNOCDAS_BINARY=$<TARGET_FILE:NoCDASim>
        -DNOCDAS_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DNOCDAS_MODE=${NOCDAS_RUN_MODE}
        -P ${CMAKE_SOURCE_DIR}/cmake/run_workload.cmake
    USES_TERMINAL
    VERBATIM
)
add_dependencies(run_workload NoCDASim)

option(NOCDAS_BUILD_TESTS "Build traffic and wait counter regression tests" OFF)
if(NOCDAS_BUILD_TESTS)
    enable_testing()
    file(GLOB NOC_SOURCES CONFIGURE_DEPENDS "src/NoC/*.cpp")
    add_executable(NoCTrafficTest tests/noc_traffic.cpp ${NOC_SOURCES})
    target_include_directories(NoCTrafficTest PRIVATE src src/NoC)
    if(CNOC_MODE)
        target_compile_definitions(NoCTrafficTest PRIVATE cNoC_MODE)
    endif()
    add_test(NAME noc_traffic COMMAND NoCTrafficTest)

    add_executable(WaitCyclesTest tests/wait_cycles.cpp src/MAC.cpp src/MACnet.cpp src/Model.cpp ${NOC_SOURCES})
    target_include_directories(WaitCyclesTest PRIVATE src src/NoC)
    if(CNOC_MODE)
        target_compile_definitions(WaitCyclesTest PRIVATE cNoC_MODE)
    endif()
    add_test(NAME wait_cycles COMMAND WaitCyclesTest)

    add_executable(KVEvictionTest tests/kv_eviction.cpp ${NOC_SOURCES})
    target_include_directories(KVEvictionTest PRIVATE src src/NoC)
    if(CNOC_MODE)
        target_compile_definitions(KVEvictionTest PRIVATE cNoC_MODE)
    endif()
    add_test(NAME kv_eviction COMMAND KVEvictionTest)
endif()
