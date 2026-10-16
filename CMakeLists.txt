cmake_minimum_required(VERSION 3.22)
project(lccsdt VERSION 0.1.0 LANGUAGES CXX)

option(LCCSDT_MPI "Enable multi-process execution" OFF)
option(LCCSDT_TESTS "Build numerical equation tests" ON)
option(LCCSDT_WRAPPER_BLAS "Use BLAS/LAPACK supplied by the compiler wrapper" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BLA_SIZEOF_INTEGER 4)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Libint2 CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED)
find_package(OpenMP QUIET COMPONENTS CXX)
set(integral_target "")
foreach(candidate IN ITEMS Libint2::int2 Libint2::libint2 libint2::int2 libint2::libint2)
  if(TARGET ${candidate})
    set(integral_target ${candidate})
    break()
  endif()
endforeach()
if(NOT integral_target)
  message(FATAL_ERROR "The integral library exports no supported CMake target")
endif()

# The molecular driver also references domain-analysis helpers. The public
# launcher exposes only the complete-space model.
file(GLOB core_sources CONFIGURE_DEPENDS "src/*.cpp")
set(reference_sources
  basis/libint2_basis_builder.cpp integrals/libint2_provider.cpp
  linalg/matrix.cpp linalg/eigensolver.cpp linalg/orthogonalization.cpp
  localization/orbital_localization.cpp localization/pipek_mezey.cpp
  localization/pipek_mezey_ciah.cpp mp2/laplace_fit.cpp
  molecule/molecule.cpp scf/rhf.cpp)
list(TRANSFORM reference_sources PREPEND "vendor/reference/src/")
add_library(lccsdt_core STATIC ${core_sources} ${reference_sources})
target_include_directories(lccsdt_core PUBLIC include vendor/reference/include)
target_link_libraries(lccsdt_core PUBLIC Eigen3::Eigen ${integral_target} Boost::headers)
if(NOT LCCSDT_WRAPPER_BLAS)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  target_link_libraries(lccsdt_core PUBLIC BLAS::BLAS LAPACK::LAPACK)
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(lccsdt_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(lccsdt_core PUBLIC MODERNQC_HAS_OPENMP=1)
endif()
set(mpi_json false)
if(LCCSDT_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_link_libraries(lccsdt_core PUBLIC MPI::MPI_CXX)
  target_compile_definitions(lccsdt_core PUBLIC LOCALIZED_CC_ENABLE_MPI=1
    MODERNQC_HAS_MPI=1 MPICH_SKIP_MPICXX=1 OMPI_SKIP_MPICXX=1)
  set(mpi_json true)
endif()
add_executable(lccsdt_engine apps/localized_ccsd_t.cpp)
target_link_libraries(lccsdt_engine PRIVATE lccsdt_core)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/lccsdt-build.json"
  CONTENT "{\"mpi\":${mpi_json},\"version\":\"${PROJECT_VERSION}\"}\n")

if(LCCSDT_TESTS)
  enable_testing()
  foreach(test IN ITEMS ccsd laplace_triples semidirect_hamiltonian pm_symmetry)
    add_executable(${test}_tests tests/test_${test}.cpp)
    target_link_libraries(${test}_tests PRIVATE lccsdt_core)
    target_compile_definitions(${test}_tests PRIVATE
      CCSD_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
      PM_SYMMETRY_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME ${test} COMMAND ${test}_tests)
  endforeach()
  add_executable(pm_symmetry_triples_tests tests/test_pm_symmetry.cpp)
  target_link_libraries(pm_symmetry_triples_tests PRIVATE lccsdt_core)
  target_compile_definitions(pm_symmetry_triples_tests PRIVATE
    PM_SYMMETRY_TRIPLES_TEST=1 PM_SYMMETRY_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME pm_symmetry_triples COMMAND pm_symmetry_triples_tests)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME launcher_contract COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_launcher.py)
  endif()
endif()
