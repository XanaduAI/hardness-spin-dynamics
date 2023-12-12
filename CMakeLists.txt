project(SpinDynamicsAnticoncentration LANGUAGES CXX)
cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "" FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

option(ENABLE_PYTHON "Enable python binding" OFF)
option(ENABLE_WARNINGS "Enable all warnings" ON)
option(BUILD_TESTS OFF)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

################################################################################
#                              Set dependencies
################################################################################

add_library(spin_dynamics_external_libs INTERFACE)
add_library(spin_dynamics_compile_flags INTERFACE)

include(FetchContent)

FetchContent_Declare(
    ExactDiagonalization
    GIT_REPOSITORY https://github.com/chaeyeunpark/ExactDiagonalization.git
    GIT_TAG master
)
FetchContent_MakeAvailable(ExactDiagonalization)

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

if(ENABLE_PYTHON)
    find_package(Python COMPONENTS Interpreter Development)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
else()
    FetchContent_Declare(
        PLKokkos
        GIT_REPOSITORY https://github.com/PennyLaneAI/pennylane-lightning-kokkos.git
        GIT_TAG v0.32.0
    )

    FetchContent_MakeAvailable(PLKokkos)

    find_package(OpenMP REQUIRED)
    find_package(Eigen3 REQUIRED)
    find_package(BLAS REQUIRED)

    target_link_libraries(spin_dynamics_external_libs INTERFACE Eigen3::Eigen BLAS::BLAS)
    target_compile_options(spin_dynamics_compile_flags INTERFACE -DEIGEN_USE_BLAS)
endif()

if(ENABLE_WARNINGS)
    target_compile_options(spin_dynamics_compile_flags INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()

if(ENABLE_PYTHON)
    pybind11_add_module(xx_hamiltonian_solver src/binding.cpp)
    target_link_libraries(xx_hamiltonian_solver PUBLIC spin_dynamics_compile_flags spin_dynamics_external_libs EDLIB::edlib)
    set_target_properties(xx_hamiltonian_solver PROPERTIES CXX_VISIBILITY_PRESET hidden)
    target_compile_features(xx_hamiltonian_solver PUBLIC cxx_std_20)
else()
    add_executable(xx_time_evol_exact src/xx_time_evol_exact.cpp)
    target_link_libraries(xx_time_evol_exact PRIVATE MPI::MPI_CXX spin_dynamics_compile_flags spin_dynamics_external_libs EDLIB::edlib)
    target_compile_features(xx_time_evol_exact PUBLIC cxx_std_20)

    add_executable(xx_time_evol_trotter src/xx_time_evol_trotter.cpp)
    target_link_libraries(xx_time_evol_trotter PUBLIC MPI::MPI_CXX pennylane_lightning_kokkos spin_dynamics_compile_flags spin_dynamics_external_libs)
    target_compile_features(xx_time_evol_trotter PUBLIC cxx_std_17)

    add_executable(xx_at_log_time_trotter src/xx_at_log_time_trotter.cpp)
    target_link_libraries(xx_at_log_time_trotter PUBLIC MPI::MPI_CXX pennylane_lightning_kokkos spin_dynamics_compile_flags spin_dynamics_external_libs)
    target_compile_features(xx_at_log_time_trotter PUBLIC cxx_std_17)

    add_executable(ising_time_evol src/ising_time_evol.cpp)
    target_link_libraries(ising_time_evol PUBLIC MPI::MPI_CXX pennylane_lightning_kokkos spin_dynamics_compile_flags spin_dynamics_external_libs)
    target_compile_features(ising_time_evol PUBLIC cxx_std_17)

    add_executable(ising_at_log_time src/ising_at_log_time.cpp)
    target_link_libraries(ising_at_log_time PUBLIC MPI::MPI_CXX pennylane_lightning_kokkos spin_dynamics_compile_flags spin_dynamics_external_libs)
    target_compile_features(ising_at_log_time PUBLIC cxx_std_17)
endif()

