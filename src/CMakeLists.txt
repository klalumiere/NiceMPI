if((GTEST_FOUND) AND ("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}/src"))
    add_executable(NiceMPIunitTests
        MPIcommunicatorHandle_tests.cpp
        NiceMPI_tests.cpp
        NiceMPIexception_tests.cpp
        tests_main.cpp
    )
    target_include_directories(NiceMPIunitTests PUBLIC ${GTEST_INCLUDE_DIRS})
   
    set_target_properties(NiceMPIunitTests PROPERTIES CXX_STANDARD 11)
    set_target_properties(NiceMPIunitTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    target_compile_options(NiceMPIunitTests PUBLIC -Wall -Wextra -pedantic)
    target_compile_options(NiceMPIunitTests PUBLIC ${MPI_CXX_COMPILE_FLAGS})
    
    string(STRIP "${MPI_CXX_LINK_FLAGS}" StripMPIflags)
    target_link_libraries(NiceMPIunitTests PUBLIC "${StripMPIflags}")
    target_link_libraries(NiceMPIunitTests PUBLIC ${MPI_CXX_LIBRARIES})
    target_link_libraries(NiceMPIunitTests PUBLIC GTest::GTest)
    add_test(NAME NiceMPIunitTests COMMAND $<TARGET_FILE:NiceMPIunitTests>)

    add_custom_target(ParallelTestsNiceMPI DEPENDS NiceMPIunitTests)
    include(ProcessorCount)
    ProcessorCount(CORES_NUMBER)
    if(CORES_NUMBER EQUAL 0)
        set(CORES_NUMBER 1)
    endif()
    set(comments "Running tests with MPI on ${CORES_NUMBER} cores. Silence means success.\n")
    set(comments "${comments}See parallelTestsStdout.txt and parallelTestsStderr.txt for detailed output.\n")
    list(APPEND argsParallel ${MPIEXEC_NUMPROC_FLAG} ${CORES_NUMBER})
    list(APPEND argsParallel ${MPIEXEC_PREFLAGS} $<TARGET_FILE:NiceMPIunitTests> ${MPIEXEC_POSTFLAGS})
    list(APPEND argsParallel 1> ${PROJECT_BINARY_DIR}/parallelTestsStdout.txt)
    list(APPEND argsParallel 2> ${PROJECT_BINARY_DIR}/parallelTestsStderr.txt)
    add_custom_command(COMMENT ${comments} COMMAND ${MPIEXEC} ARGS ${argsParallel} TARGET ParallelTestsNiceMPI)
    add_test(NAME ParallelTestsNiceMPI COMMAND ${MPIEXEC} ${argsParallel})

    configure_file(buildInformationNiceMPI.h.in ${PROJECT_SOURCE_DIR}/src/buildInformationNiceMPI.h @ONLY)
endif()
