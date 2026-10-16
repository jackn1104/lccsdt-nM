#include "localized_cc/cuda_triples.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace localized_cc {
namespace {

extern "C" {
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc);
}

constexpr std::array<std::array<std::size_t, 3>, 6> permutations{{
    {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
    {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}}}};
constexpr std::array<double, 6> permutation_sign{{1.0, -1.0, -1.0,
                                                  1.0, 1.0, -1.0}};

std::size_t checked_product(const std::size_t left,
                            const std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error("triples moment tile size overflows");
  return left * right;
}

void add_elements(std::size_t& total, const std::size_t count) {
  if (count > std::numeric_limits<std::size_t>::max() - total)
    throw std::overflow_error("triples moment workspace size overflows");
  total += count;
}

bool finite_vector(const std::vector<double>& values) {
  for (const double value : values)
    if (!std::isfinite(value))
      return false;
  return true;
}

int checked_blas_dimension(const std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("contraction tile exceeds BLAS integer range");
  return static_cast<int>(value);
}

void require_size(const std::vector<double>& values,
                  const std::size_t expected) {
  if (values.size() != expected)
    throw std::invalid_argument(
        "triples moment tile packed dimensions are inconsistent");
  if (!finite_vector(values))
    throw std::invalid_argument(
        "triples moment tile contains non-finite packed values");
}

std::array<std::size_t, 6>
decode_offset(std::size_t offset,
              const std::array<std::size_t, 6>& dimensions) {
  std::array<std::size_t, 6> indices{};
  for (std::size_t axis = dimensions.size(); axis-- > 0;) {
    indices[axis] = offset % dimensions[axis];
    offset /= dimensions[axis];
  }
  return indices;
}

} // namespace

std::size_t triples_energy_tile_size(const TriplesEnergyTile& tile) {
  std::size_t size = 1;
  for (const std::size_t dimension : tile.dimensions) {
    if (dimension == 0)
      throw std::invalid_argument(
          "triples energy tile dimensions must be nonzero");
    if (size > std::numeric_limits<std::size_t>::max() / dimension)
      throw std::overflow_error("triples energy tile size overflows");
    size *= dimension;
  }
  return size;
}

void validate_triples_energy_tile(const TriplesEnergyTile& tile) {
  const std::size_t size = triples_energy_tile_size(tile);
  if (tile.doubles_moment.size() != size ||
      tile.singles_moment.size() != size)
    throw std::invalid_argument(
        "triples energy tile moment dimensions are inconsistent");
  for (std::size_t axis = 0; axis < tile.dimensions.size(); ++axis)
    if (tile.orbital_energies[axis].size() != tile.dimensions[axis])
      throw std::invalid_argument(
          "triples energy tile orbital-energy dimensions are inconsistent");
  if (!std::isfinite(tile.symmetry_factor) ||
      !(tile.denominator_floor > 0.0) ||
      !std::isfinite(tile.denominator_floor) ||
      !finite_vector(tile.doubles_moment) ||
      !finite_vector(tile.singles_moment))
    throw std::invalid_argument("triples energy tile contains invalid values");
  for (const auto& energies : tile.orbital_energies)
    if (!finite_vector(energies))
      throw std::invalid_argument(
          "triples energy tile contains non-finite orbital energies");
}

TriplesEnergyPair reduce_triples_energy_cpu(const TriplesEnergyTile& tile) {
  validate_triples_energy_tile(tile);
  const std::size_t size = triples_energy_tile_size(tile);
  TriplesEnergyPair result;
  for (std::size_t offset = 0; offset < size; ++offset) {
    const auto index = decode_offset(offset, tile.dimensions);
    const double denominator =
        tile.orbital_energies[0][index[0]] +
        tile.orbital_energies[1][index[1]] +
        tile.orbital_energies[2][index[2]] -
        tile.orbital_energies[3][index[3]] -
        tile.orbital_energies[4][index[4]] -
        tile.orbital_energies[5][index[5]];
    if (!std::isfinite(denominator) ||
        std::abs(denominator) < tile.denominator_floor)
      throw std::runtime_error("small or invalid triples tile denominator");
    const double doubles = tile.doubles_moment[offset];
    const double singles = tile.singles_moment[offset];
    result.bracket +=
        tile.symmetry_factor * doubles * doubles / denominator;
    result.parenthetical += tile.symmetry_factor * doubles *
                            (doubles + singles) / denominator;
  }
  return result;
}

void validate_triples_moment_tile(const TriplesMomentTile& tile,
                                  const bool allow_shared_pairs) {
  if (tile.occupied_count < 3 || tile.virtual_count < 3 ||
      !(tile.denominator_floor > 0.0) ||
      !std::isfinite(tile.denominator_floor))
    throw std::invalid_argument("invalid triples moment tile dimensions");
  for (const double energy : tile.occupied_energies)
    if (!std::isfinite(energy))
      throw std::invalid_argument(
          "triples moment tile contains non-finite occupied energies");
  if ((tile.virtual_stride != 1 && tile.virtual_stride != 2) ||
      (tile.virtual_stride != 1 && !allow_shared_pairs))
    throw std::invalid_argument("unsupported triples virtual stride (CPU only)");
  if(tile.exact_spin_channels && (!allow_shared_pairs || tile.virtual_stride!=2 ||
      std::any_of(tile.occupied_spins.begin(),tile.occupied_spins.end(),
                  [](auto spin){return spin>1;})))
    throw std::invalid_argument("invalid exact triples spin-channel metadata");
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const auto dimension = tile.virtual_dimensions[axis];
    if (dimension == 0 || tile.virtual_offsets[axis] >= tile.virtual_count ||
        dimension - 1 > (tile.virtual_count - 1 - tile.virtual_offsets[axis]) / tile.virtual_stride ||
        tile.virtual_energies[axis].size() != dimension ||
        !finite_vector(tile.virtual_energies[axis]))
      throw std::invalid_argument(
          "invalid triples moment tile virtual range or energies");
  }

  for (std::size_t op = 0; op < tile.occupied_permutations.size(); ++op) {
    const auto source = tile.pair_operand_sources[op];
    if (source >= tile.occupied_permutations.size() ||
        tile.pair_operand_sources[source] != source ||
        (!allow_shared_pairs && source != op))
      throw std::invalid_argument("invalid or unsupported triples pair operand alias");
    const auto& occupied_permutation = tile.occupied_permutations[op];
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const auto dimension = tile.virtual_dimensions[axis];
      require_size(occupied_permutation.t2_jk_axis_e[axis],
                   checked_product(dimension, tile.virtual_count));
      require_size(occupied_permutation.eri_m_axis[axis],
                   checked_product(tile.occupied_count, dimension));
      require_size(occupied_permutation.t1_axis[axis], dimension);
      require_size(occupied_permutation.fock_axis[axis], dimension);
    }
    for (std::size_t virtual_permutation = 0;
         virtual_permutation < permutations.size(); ++virtual_permutation) {
      const auto& permutation = permutations[virtual_permutation];
      const auto b_dimension = tile.virtual_dimensions[permutation[1]];
      const auto c_dimension = tile.virtual_dimensions[permutation[2]];
      const auto pair_size = checked_product(b_dimension, c_dimension);
      require_size(occupied_permutation.eri_pair_e[virtual_permutation],
                   source == op ? checked_product(pair_size, tile.virtual_count) : 0);
      require_size(occupied_permutation.t2_m_pair[virtual_permutation],
                   source == op ? checked_product(tile.occupied_count, pair_size) : 0);
      require_size(occupied_permutation.eri_pair_jk[virtual_permutation],
                   pair_size);
    }
  }
}

TriplesMomentResult
contract_triples_moment_cpu(const TriplesMomentTile& tile) {
  validate_triples_moment_tile(tile, true);
  TriplesMomentResult result;
  const auto tile_size = checked_product(
      checked_product(tile.virtual_dimensions[0],
                      tile.virtual_dimensions[1]),
      tile.virtual_dimensions[2]);
  result.doubles_moment.assign(tile_size, 0.0);
  result.left_moment.assign(tile_size, 0.0);
  std::size_t workspace_elements = 0;
  add_elements(workspace_elements, tile_size);
  add_elements(workspace_elements, tile_size);
  for (const auto& energies : tile.virtual_energies)
    add_elements(workspace_elements, energies.size());
  for (const auto& data : tile.occupied_permutations) {
    for (const auto& values : data.t2_jk_axis_e)
      add_elements(workspace_elements, values.size());
    for (const auto& values : data.eri_pair_e)
      add_elements(workspace_elements, values.size());
    for (const auto& values : data.t2_m_pair)
      add_elements(workspace_elements, values.size());
    for (const auto& values : data.eri_m_axis)
      add_elements(workspace_elements, values.size());
    for (const auto& values : data.t1_axis)
      add_elements(workspace_elements, values.size());
    for (const auto& values : data.eri_pair_jk)
      add_elements(workspace_elements, values.size());
    for (const auto& values : data.fock_axis)
      add_elements(workspace_elements, values.size());
  }
  constexpr std::size_t task_count =
      permutations.size() * permutations.size();
  const auto task_elements = checked_product(task_count, tile_size);
  std::vector<double> task_doubles(task_elements, 0.0);
  std::vector<double> task_singles(task_elements, 0.0);
  add_elements(workspace_elements, task_doubles.size());
  add_elements(workspace_elements, task_singles.size());
  result.workspace_bytes =
      checked_product(workspace_elements, sizeof(double));

  // Each P(ijk)P(abc) task contains two dense contractions,
  //
  //   R(a,bc) = T2(a,e) G(bc,e) - G(m,a) T2(m,bc).
  //
  // The previous CPU path evaluated both dot products separately for every
  // (a,b,c), leaving serial BLAS builds almost entirely idle.  Form the full
  // bounded a-by-(bc) panel with two DGEMMs, then scatter only globally
  // ordered a<b<c entries.  Task-private panels avoid atomics and the final
  // fixed-order reduction preserves deterministic permutation summation.
  // Subdivide each permutation by independent external-a rows. At width 64
  // this exposes 288 tasks instead of capping useful workers at 36.
  constexpr std::size_t external_block = 8;
  const auto maximum_dimension = *std::max_element(
      tile.virtual_dimensions.begin(), tile.virtual_dimensions.end());
  const auto slices = (maximum_dimension + external_block - 1) / external_block;
  // Adjacent virtual permutations retain a and exchange b,c. The integral
  // and permutation signs both reverse, so their signed virtual-sum
  // contributions coincide at the SAME external tuple. Do not fold the
  // occupied sum or singles: this optimization assumes no T2 symmetry.
  if (tile.antisymmetric_virtual_integrals) {
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static) if(tile_size >= 64)
#endif
    for (std::size_t half_task=0; half_task<task_count/2; ++half_task)
    for (std::size_t slice=0; slice<slices; ++slice) {
      const auto task=2*half_task, op=task/6, vp=task%6;
      const auto& p=permutations[vp];
      const auto aa=p[0], bb=p[1], cc=p[2];
      const auto na=tile.virtual_dimensions[aa], nb=tile.virtual_dimensions[bb], nc=tile.virtual_dimensions[cc];
      const auto first=slice*external_block;
      if(first>=na) continue;
      const int e_spin=static_cast<int>(tile.virtual_offsets[bb]%2+tile.virtual_offsets[cc]%2)
          -static_cast<int>(tile.occupied_spins[permutations[op][0]]);
      if(tile.exact_spin_channels && (e_spin<0 || e_spin>1)) continue;
      const auto count=std::min(external_block,na-first), np=checked_product(nb,nc);
      std::vector<double> raw(checked_product(count,np));
      const int pairs=checked_blas_dimension(np), external=checked_blas_dimension(count), nv=checked_blas_dimension(tile.virtual_count);
      const char trans='T', normal='N'; const double one=1.0, zero=0.0;
      const auto& data=tile.occupied_permutations[op];
      const auto& pair=tile.occupied_permutations[tile.pair_operand_sources[op]];
      dgemm_(&trans,&normal,&pairs,&external,&nv,&one,pair.eri_pair_e[vp].data(),&nv,
             data.t2_jk_axis_e[aa].data()+first*tile.virtual_count,&nv,&zero,raw.data(),&pairs);
      const double factor=0.25*permutation_sign[op]*permutation_sign[vp];
      for(std::size_t a=first;a<first+count;++a)
      for(std::size_t b=0;b<nb;++b)
      for(std::size_t c=0;c<nc;++c) {
        std::array<std::size_t,3> x{}; x[aa]=a; x[bb]=b; x[cc]=c;
        const auto g0=tile.virtual_offsets[0]+tile.virtual_stride*x[0];
        const auto g1=tile.virtual_offsets[1]+tile.virtual_stride*x[1];
        const auto g2=tile.virtual_offsets[2]+tile.virtual_stride*x[2];
        if(!(g0<g1 && g1<g2)) continue;
        const auto index=(x[0]*tile.virtual_dimensions[1]+x[1])*tile.virtual_dimensions[2]+x[2];
        const auto value=factor*raw[(a-first)*np+b*nc+c];
        task_doubles[task*tile_size+index]=value;
        task_doubles[(task+1)*tile_size+index]=value;
      }
    }
  } // implicit OpenMP barrier before occupied sums update the same arrays
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static) if(tile_size >= 64)
#endif
  for (std::ptrdiff_t raw_task = 0;
       raw_task < static_cast<std::ptrdiff_t>(task_count); ++raw_task)
  for (std::size_t slice = 0; slice < slices; ++slice) {
    const auto task = static_cast<std::size_t>(raw_task);
    const auto occupied_permutation = task / permutations.size();
    const auto virtual_permutation = task % permutations.size();
    const auto& data = tile.occupied_permutations[occupied_permutation];
    const auto& pair_data = tile.occupied_permutations[
        tile.pair_operand_sources[occupied_permutation]];
    const auto& permutation = permutations[virtual_permutation];
    const auto a_axis = permutation[0];
    const auto b_axis = permutation[1];
    const auto c_axis = permutation[2];
    const auto a_dimension = tile.virtual_dimensions[a_axis];
    const auto b_dimension = tile.virtual_dimensions[b_axis];
    const auto c_dimension = tile.virtual_dimensions[c_axis];
    const auto pair_size = checked_product(b_dimension, c_dimension);
    const auto a_first = slice * external_block;
    if (a_first >= a_dimension) continue;
    const auto a_count = std::min(external_block, a_dimension - a_first);
    std::vector<double> raw(checked_product(a_count, pair_size), 0.0);
    const int pairs = checked_blas_dimension(pair_size);
    const int external = checked_blas_dimension(a_count);
    const int axis_stride = checked_blas_dimension(a_dimension);
    const int virtuals = checked_blas_dimension(tile.virtual_count);
    const int occupied = checked_blas_dimension(tile.occupied_count);
    constexpr char transpose = 'T';
    constexpr char no_transpose = 'N';
    constexpr double one = 1.0;
    constexpr double minus_one = -1.0;
    constexpr double zero = 0.0;
    // Spin-diagonal ERIs require e_beta=b_beta+c_beta-i_beta and
    // m_beta=j_beta+k_beta-a_beta. Values outside {0,1} make the
    // entire corresponding integral operand exactly zero. Singles remain.
    bool virtual_channel=true, occupied_channel=true;
    if(tile.exact_spin_channels) {
      const auto& op=permutations[occupied_permutation];
      const int i=static_cast<int>(tile.occupied_spins[op[0]]);
      const int j=static_cast<int>(tile.occupied_spins[op[1]]);
      const int k=static_cast<int>(tile.occupied_spins[op[2]]);
      const int a=static_cast<int>(tile.virtual_offsets[a_axis]%2);
      const int b=static_cast<int>(tile.virtual_offsets[b_axis]%2);
      const int c=static_cast<int>(tile.virtual_offsets[c_axis]%2);
      const int e=b+c-i, m=j+k-a;
      virtual_channel=e>=0 && e<=1;
      occupied_channel=m>=0 && m<=1;
    }
    if(virtual_channel && !tile.antisymmetric_virtual_integrals) dgemm_(&transpose, &no_transpose, &pairs, &external, &virtuals, &one,
           pair_data.eri_pair_e[virtual_permutation].data(), &virtuals,
           data.t2_jk_axis_e[a_axis].data() + a_first * tile.virtual_count, &virtuals, &zero, raw.data(),
           &pairs);
    if(occupied_channel) dgemm_(&no_transpose, &transpose, &pairs, &external, &occupied,
           &minus_one, pair_data.t2_m_pair[virtual_permutation].data(), &pairs,
           data.eri_m_axis[a_axis].data() + a_first, &axis_stride, &one, raw.data(),
           &pairs);
    const double factor =
        0.25 * permutation_sign[occupied_permutation] *
        permutation_sign[virtual_permutation];
    for (std::size_t a = a_first; a < a_first + a_count; ++a)
      for (std::size_t b = 0; b < b_dimension; ++b)
        for (std::size_t c = 0; c < c_dimension; ++c) {
        std::array<std::size_t, 3> local{};
        local[a_axis] = a;
        local[b_axis] = b;
        local[c_axis] = c;
        const auto x0 = local[0], x1 = local[1], x2 = local[2];
        const std::array<std::size_t, 3> global{
            tile.virtual_offsets[0] + tile.virtual_stride * x0,
            tile.virtual_offsets[1] + tile.virtual_stride * x1,
            tile.virtual_offsets[2] + tile.virtual_stride * x2};
        if (!(global[0] < global[1] && global[1] < global[2]))
          continue;
        const auto pair = b * c_dimension + c;
        const auto linear =
            (x0 * tile.virtual_dimensions[1] + x1) *
                tile.virtual_dimensions[2] +
            x2;
        task_doubles[task * tile_size + linear] +=
            factor * raw[(a - a_first) * pair_size + pair];
        task_singles[task * tile_size + linear] =
            factor *
            (data.t1_axis[a_axis][a] *
                 data.eri_pair_jk[virtual_permutation][pair] +
             data.fock_axis[a_axis][a] *
                 data.t2_jk_axis_e[b_axis]
                     [b * tile.virtual_count + global[c_axis]]);
      }
  }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(tile_size >= 4096)
#endif
  for (std::ptrdiff_t raw_linear = 0;
       raw_linear < static_cast<std::ptrdiff_t>(tile_size); ++raw_linear) {
    const auto linear = static_cast<std::size_t>(raw_linear);
    double doubles = 0.0;
    double singles = 0.0;
    for (std::size_t task = 0; task < task_count; ++task) {
      doubles += task_doubles[task * tile_size + linear];
      singles += task_singles[task * tile_size + linear];
    }
    result.doubles_moment[linear] = doubles;
    result.left_moment[linear] = doubles + singles;
  }

  for (std::size_t x0 = 0; x0 < tile.virtual_dimensions[0]; ++x0)
    for (std::size_t x1 = 0; x1 < tile.virtual_dimensions[1]; ++x1)
      for (std::size_t x2 = 0; x2 < tile.virtual_dimensions[2]; ++x2) {
        const std::array<std::size_t, 3> global{
            tile.virtual_offsets[0] + tile.virtual_stride * x0,
            tile.virtual_offsets[1] + tile.virtual_stride * x1,
            tile.virtual_offsets[2] + tile.virtual_stride * x2};
        if (!(global[0] < global[1] && global[1] < global[2]))
          continue;
        const double denominator =
            tile.occupied_energies[0] + tile.occupied_energies[1] +
            tile.occupied_energies[2] - tile.virtual_energies[0][x0] -
            tile.virtual_energies[1][x1] -
            tile.virtual_energies[2][x2];
        if (!std::isfinite(denominator) ||
            std::abs(denominator) < tile.denominator_floor)
          throw std::runtime_error(
              "small or invalid triples moment tile denominator");
        const auto linear =
            (x0 * tile.virtual_dimensions[1] + x1) *
                tile.virtual_dimensions[2] +
            x2;
        const auto doubles = result.doubles_moment[linear];
        result.parenthetical +=
            doubles * result.left_moment[linear] / denominator;
        ++result.evaluated_virtual_triples;
      }
  return result;
}

void validate_ccsd_wabef_tile(const CcsdWabefTile& tile) {
  if (tile.occupied_pairs == 0 || tile.virtual_pair_elements == 0 ||
      tile.virtual_pair_tasks == 0)
    throw std::invalid_argument("invalid CCSD W_abef tile dimensions");
  const auto tau_size =
      checked_product(tile.occupied_pairs, tile.virtual_pair_elements);
  const auto wabef_size =
      checked_product(tile.virtual_pair_tasks, tile.virtual_pair_elements);
  if (tile.tau.size() != tau_size || tile.wabef.size() != wabef_size ||
      !finite_vector(tile.tau) || !finite_vector(tile.wabef))
    throw std::invalid_argument(
        "CCSD W_abef tile packed dimensions are inconsistent");
}

CcsdWabefResult contract_ccsd_wabef_cpu(const CcsdWabefTile& tile) {
  validate_ccsd_wabef_tile(tile);
  CcsdWabefResult result;
  result.contractions.resize(
      checked_product(tile.virtual_pair_tasks, tile.occupied_pairs));
  // Row-major tau(P,K) and W(T,K) are column-major views tau(K,P) and
  // W(K,T). C(P,T)=tau^T*W therefore has the desired row-major T-by-P layout.
  const auto pairs = checked_blas_dimension(tile.occupied_pairs);
  const auto contraction = checked_blas_dimension(tile.virtual_pair_elements);
  constexpr char transpose = 'T';
  constexpr char no_transpose = 'N';
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;
#if defined(_OPENMP)
  const auto workers = std::min(
      tile.virtual_pair_tasks,
      static_cast<std::size_t>(omp_get_max_threads()));
#endif
  // The NERSC build deliberately links the serial LibSci BLAS interface so
  // unrelated BLAS calls cannot oversubscribe a hybrid MPI/OpenMP job.  A
  // single DGEMM would consequently use only one core.  Split the independent
  // external-(a,b) columns into contiguous slabs and let each OpenMP worker
  // call serial DGEMM for its own slab.  Columns are disjoint, so this changes
  // execution ownership only; the CCSD W_abef equation and reduction order
  // within every output element are unchanged.
#if defined(_OPENMP)
#pragma omp parallel num_threads(static_cast<int>(workers))
#endif
  {
    std::size_t worker = 0;
    std::size_t active_workers = 1;
#if defined(_OPENMP)
    worker = static_cast<std::size_t>(omp_get_thread_num());
    active_workers = static_cast<std::size_t>(omp_get_num_threads());
#endif
    const auto first = tile.virtual_pair_tasks * worker / active_workers;
    const auto last =
        tile.virtual_pair_tasks * (worker + 1) / active_workers;
    const auto local_tasks = checked_blas_dimension(last - first);
    if (local_tasks != 0)
      dgemm_(&transpose, &no_transpose, &pairs, &local_tasks, &contraction,
             &alpha, tile.tau.data(), &contraction,
             tile.wabef.data() +
                 static_cast<std::ptrdiff_t>(first *
                                             tile.virtual_pair_elements),
             &contraction, &beta,
             result.contractions.data() +
                 static_cast<std::ptrdiff_t>(first * tile.occupied_pairs),
             &pairs);
  }
  if (!finite_vector(result.contractions))
    throw std::runtime_error(
        "CPU BLAS CCSD W_abef contraction produced a non-finite value");
  result.workspace_bytes =
      result.contractions.size() * sizeof(double);
  return result;
}

} // namespace localized_cc
