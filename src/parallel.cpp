#include "localized_cc/parallel.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#if defined(LOCALIZED_CC_ENABLE_MPI)
#include <mpi.h>
#endif

namespace localized_cc::parallel {
namespace {

#if defined(LOCALIZED_CC_ENABLE_MPI)
bool mpi_active() {
  int initialized = 0;
  int finalized = 0;
  MPI_Initialized(&initialized);
  if (initialized != 0)
    MPI_Finalized(&finalized);
  return initialized != 0 && finalized == 0;
}

void require_mpi_success(const int status, const char* operation) {
  if (status != MPI_SUCCESS)
    throw std::runtime_error(operation);
}
#endif

} // namespace

Runtime::Runtime(int& argc, char**& argv) {
#if defined(LOCALIZED_CC_ENABLE_MPI)
  int initialized = 0;
  require_mpi_success(MPI_Initialized(&initialized),
                      "MPI_Initialized failed");
  if (initialized == 0) {
    int provided = MPI_THREAD_SINGLE;
    require_mpi_success(
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided),
        "MPI_Init_thread failed");
    if (provided < MPI_THREAD_FUNNELED) {
      MPI_Abort(MPI_COMM_WORLD, 2);
      std::abort();
    }
    owns_runtime_ = true;
  }
#else
  (void)argc;
  (void)argv;
#endif
}

Runtime::~Runtime() {
#if defined(LOCALIZED_CC_ENABLE_MPI)
  if (owns_runtime_ && mpi_active())
    MPI_Finalize();
#endif
}

World Runtime::world() const { return parallel::world(); }

[[noreturn]] void Runtime::abort(const int error_code) const {
#if defined(LOCALIZED_CC_ENABLE_MPI)
  if (mpi_active())
    MPI_Abort(MPI_COMM_WORLD, error_code);
#endif
  std::exit(error_code);
}

World world() {
#if defined(LOCALIZED_CC_ENABLE_MPI)
  if (mpi_active()) {
    int rank = 0;
    int size = 1;
    require_mpi_success(MPI_Comm_rank(MPI_COMM_WORLD, &rank),
                        "MPI_Comm_rank failed");
    require_mpi_success(MPI_Comm_size(MPI_COMM_WORLD, &size),
                        "MPI_Comm_size failed");
    if (rank < 0 || size < 1)
      throw std::runtime_error("invalid MPI world dimensions");
    return {static_cast<std::size_t>(rank), static_cast<std::size_t>(size)};
  }
#endif
  return {};
}

void sum_in_place(std::vector<double>& values) {
#if defined(LOCALIZED_CC_ENABLE_MPI)
  if (!mpi_active() || world().size == 1)
    return;
  constexpr auto maximum_count =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  std::size_t offset = 0;
  while (offset < values.size()) {
    const auto count = std::min(maximum_count, values.size() - offset);
    require_mpi_success(
        MPI_Allreduce(MPI_IN_PLACE, values.data() + offset,
                      static_cast<int>(count), MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD),
        "MPI_Allreduce failed");
    offset += count;
  }
#else
  (void)values;
#endif
}

void transfer_tile(std::vector<double>& values, std::size_t source,
                   std::size_t destination) {
  const auto w = world();
  if (source >= w.size || destination >= w.size)
    throw std::invalid_argument("tile transfer rank out of range");
  if (source == destination) return;
#if defined(LOCALIZED_CC_ENABLE_MPI)
  if (w.rank != source && w.rank != destination) return;
  constexpr int tile_tag = 17031; // ordered adaptive-MP2 stream, no async users
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  for (std::size_t offset = 0; offset < values.size();) {
    const int count = static_cast<int>(std::min(limit, values.size()-offset));
    if (w.rank == source)
      require_mpi_success(MPI_Send(values.data()+offset, count, MPI_DOUBLE,
          static_cast<int>(destination), tile_tag, MPI_COMM_WORLD), "MP2 tile send failed");
    else
      require_mpi_success(MPI_Recv(values.data()+offset, count, MPI_DOUBLE,
          static_cast<int>(source), tile_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MP2 tile receive failed");
    offset += static_cast<std::size_t>(count);
  }
#else
  (void)values;
#endif
}

void sum_in_place_group(std::vector<double>& values,
                        const std::size_t color) {
#if defined(LOCALIZED_CC_ENABLE_MPI)
  if (!mpi_active() || world().size == 1)
    return;
  if (color > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("MPI subgroup color exceeds the integer range");
  MPI_Comm communicator = MPI_COMM_NULL;
  require_mpi_success(
      MPI_Comm_split(MPI_COMM_WORLD, static_cast<int>(color),
                     static_cast<int>(world().rank), &communicator),
      "MPI_Comm_split failed");
  try {
    constexpr auto maximum_count =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    std::size_t offset = 0;
    while (offset < values.size()) {
      const auto count = std::min(maximum_count, values.size() - offset);
      require_mpi_success(
          MPI_Allreduce(MPI_IN_PLACE, values.data() + offset,
                        static_cast<int>(count), MPI_DOUBLE, MPI_SUM,
                        communicator),
          "MPI subgroup Allreduce failed");
      offset += count;
    }
    require_mpi_success(MPI_Comm_free(&communicator),
                        "MPI_Comm_free failed");
  } catch (...) {
    if (communicator != MPI_COMM_NULL)
      MPI_Comm_free(&communicator);
    throw;
  }
#else
  (void)values;
  (void)color;
#endif
}

std::uint64_t sum_group(const std::uint64_t value,
                        const std::size_t color) {
#if defined(LOCALIZED_CC_ENABLE_MPI)
  if (!mpi_active() || world().size == 1)
    return value;
  if (color > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("MPI subgroup color exceeds the integer range");
  MPI_Comm communicator = MPI_COMM_NULL;
  require_mpi_success(
      MPI_Comm_split(MPI_COMM_WORLD, static_cast<int>(color),
                     static_cast<int>(world().rank), &communicator),
      "MPI_Comm_split failed");
  std::uint64_t result = 0;
  try {
    require_mpi_success(
        MPI_Allreduce(&value, &result, 1, MPI_UINT64_T, MPI_SUM, communicator),
        "MPI subgroup counter Allreduce failed");
    require_mpi_success(MPI_Comm_free(&communicator),
                        "MPI_Comm_free failed");
  } catch (...) {
    if (communicator != MPI_COMM_NULL)
      MPI_Comm_free(&communicator);
    throw;
  }
  return result;
#else
  (void)color;
  return value;
#endif
}

#if defined(LOCALIZED_CC_ENABLE_MPI)
template <typename T>
void broadcast_vector(std::vector<T>& values, const std::size_t root,
                      const MPI_Datatype datatype) {
  if (!mpi_active() || world().size == 1)
    return;
  if (root >= world().size)
    throw std::invalid_argument("MPI broadcast root is out of range");
  constexpr auto maximum_count =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  std::size_t offset = 0;
  while (offset < values.size()) {
    const auto count = std::min(maximum_count, values.size() - offset);
    require_mpi_success(
        MPI_Bcast(values.data() + offset, static_cast<int>(count), datatype,
                  static_cast<int>(root), MPI_COMM_WORLD),
        "MPI_Bcast failed");
    offset += count;
  }
}
#endif

void broadcast_in_place(std::vector<double>& values, const std::size_t root) {
#if defined(LOCALIZED_CC_ENABLE_MPI)
  broadcast_vector(values, root, MPI_DOUBLE);
#else
  (void)values;
  (void)root;
#endif
}

void broadcast_in_place(std::vector<int>& values, const std::size_t root) {
#if defined(LOCALIZED_CC_ENABLE_MPI)
  broadcast_vector(values, root, MPI_INT);
#else
  (void)values;
  (void)root;
#endif
}

void barrier() {
#if defined(LOCALIZED_CC_ENABLE_MPI)
  if (mpi_active() && world().size > 1)
    require_mpi_success(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier failed");
#endif
}

} // namespace localized_cc::parallel
