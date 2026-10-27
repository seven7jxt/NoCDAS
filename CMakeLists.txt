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

list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
add_library(nocdas_core STATIC ${SOURCES} ${HEADERS})
target_include_directories(nocdas_core PUBLIC src src/NoC)
target_compile_definitions(nocdas_core PRIVATE NOCDAS_BUILD_TYPE="$<CONFIG>")
set(NI_TX_PACKET_CAPACITY "" CACHE STRING "Override NI TX packets per VN (0: unlimited; empty: parameters.hpp)")
if(NOT NI_TX_PACKET_CAPACITY STREQUAL "")
    target_compile_definitions(nocdas_core PUBLIC NI_TX_PACKET_CAPACITY=${NI_TX_PACKET_CAPACITY})
endif()

add_executable(NoCDASim src/main.cpp)
target_link_libraries(NoCDASim PRIVATE nocdas_core)

if(CNOC_MODE)
    target_compile_definitions(nocdas_core PUBLIC cNoC_MODE)
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
    add_executable(CNoCOutputTilingTest tests/cnoc_output_tiling.cpp
        src/MAC.cpp src/MACnet.cpp src/Model.cpp ${NOC_SOURCES})
    target_include_directories(CNoCOutputTilingTest PRIVATE src src/NoC)
    target_compile_definitions(CNoCOutputTilingTest PRIVATE cNoC_MODE ROUTER_SRAM_LIMIT=64)
    add_test(NAME cnoc_output_tiling COMMAND CNoCOutputTilingTest)
    add_test(NAME cnoc_output_tiling_exact COMMAND CNoCOutputTilingTest exact)
    add_test(NAME cnoc_output_tiling_fallback COMMAND CNoCOutputTilingTest fallback)
    add_executable(AttentionExpPipelineTest tests/attention_exp_pipeline.cpp ${NOC_SOURCES})
    target_include_directories(AttentionExpPipelineTest PRIVATE src src/NoC)
    target_compile_definitions(AttentionExpPipelineTest PRIVATE cNoC_MODE)
    add_test(NAME attention_exp_pipeline COMMAND AttentionExpPipelineTest)
    add_executable(SramPeaksTest tests/sram_peaks.cpp
        src/MAC.cpp src/MACnet.cpp src/Model.cpp ${NOC_SOURCES})
    target_include_directories(SramPeaksTest PRIVATE src src/NoC)
    add_test(NAME sram_peaks COMMAND SramPeaksTest)
    add_executable(BaselineTerminalReferenceTest tests/cnoc_terminal.cpp
        src/MAC.cpp src/MACnet.cpp src/Model.cpp ${NOC_SOURCES})
    target_include_directories(BaselineTerminalReferenceTest PRIVATE src src/NoC)
    target_compile_definitions(BaselineTerminalReferenceTest PRIVATE PE_GATE_BATCH_PAIRS=4)
    add_test(NAME terminal_baseline_reference COMMAND BaselineTerminalReferenceTest)
    add_test(NAME terminal_baseline_geglu_reference COMMAND BaselineTerminalReferenceTest geglu)
    foreach(terminal_variant IN ITEMS Serial AbstractQueue)
        add_executable(CNoCTerminal${terminal_variant}Test tests/cnoc_terminal.cpp
            src/MAC.cpp src/MACnet.cpp src/Model.cpp ${NOC_SOURCES})
        target_include_directories(CNoCTerminal${terminal_variant}Test PRIVATE src src/NoC)
        target_compile_definitions(CNoCTerminal${terminal_variant}Test PRIVATE
            cNoC_MODE ROUTER_SRAM_LIMIT=256 PE_GATE_BATCH_PAIRS=4)
        if(terminal_variant STREQUAL "AbstractQueue")
            target_compile_definitions(CNoCTerminal${terminal_variant}Test PRIVATE
                CNOC_GATE_BATCH_PAIRS=80 MAC_INPUT_SRAM_LIMIT=64 EXPECT_ABSTRACT_QUEUE)
        endif()
        add_test(NAME cnoc_terminal_${terminal_variant} COMMAND CNoCTerminal${terminal_variant}Test)
        add_test(NAME cnoc_terminal_geglu_${terminal_variant} COMMAND CNoCTerminal${terminal_variant}Test geglu)
        if(terminal_variant STREQUAL "Serial")
            add_test(NAME cnoc_terminal_fallback_${terminal_variant} COMMAND CNoCTerminal${terminal_variant}Test fallback)
        endif()
    endforeach()
    foreach(gate_variant IN ITEMS Scalar Serial Capacity NoAck)
        add_executable(GateBatch${gate_variant}Test tests/gate_batch.cpp
            src/MAC.cpp src/MACnet.cpp src/Model.cpp ${NOC_SOURCES})
        target_include_directories(GateBatch${gate_variant}Test PRIVATE src src/NoC)
        if(gate_variant STREQUAL "Scalar")
            target_compile_definitions(GateBatch${gate_variant}Test PRIVATE PE_GATE_BATCH_PAIRS=1 PE_ADD_BATCH_PAIRS=1)
        else()
            target_compile_definitions(GateBatch${gate_variant}Test PRIVATE PE_GATE_BATCH_PAIRS=4 PE_ADD_BATCH_PAIRS=4)
        endif()
        if(gate_variant STREQUAL "Capacity")
            target_compile_definitions(GateBatch${gate_variant}Test PRIVATE MAC_INPUT_SRAM_LIMIT=6)
        endif()
        if(gate_variant STREQUAL "NoAck")
            target_compile_definitions(GateBatch${gate_variant}Test PRIVATE only3type)
        endif()
        add_test(NAME gate_swiglu_${gate_variant} COMMAND GateBatch${gate_variant}Test)
        add_test(NAME gate_geglu_${gate_variant} COMMAND GateBatch${gate_variant}Test geglu)
        add_test(NAME add_batch_${gate_variant} COMMAND GateBatch${gate_variant}Test add)
    endforeach()
    add_executable(AddBatchThroughputTest tests/gate_batch.cpp
        src/MAC.cpp src/MACnet.cpp src/Model.cpp ${NOC_SOURCES})
    target_include_directories(AddBatchThroughputTest PRIVATE src src/NoC)
    target_compile_definitions(AddBatchThroughputTest PRIVATE PE_NUM_OP=3 PE_ADD_BATCH_PAIRS=4)
    add_test(NAME add_batch_throughput COMMAND AddBatchThroughputTest add)
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

    add_executable(CNoCChunkingTest tests/cnoc_chunking.cpp src/MAC.cpp src/MACnet.cpp src/Model.cpp ${NOC_SOURCES})
    target_include_directories(CNoCChunkingTest PRIVATE src src/NoC)
    target_compile_definitions(CNoCChunkingTest PRIVATE cNoC_MODE)
    add_test(NAME cnoc_chunking COMMAND CNoCChunkingTest)
    add_test(NAME cnoc_chunking_impossible COMMAND CNoCChunkingTest impossible)
    set_tests_properties(cnoc_chunking_impossible PROPERTIES WILL_FAIL TRUE)
endif()
