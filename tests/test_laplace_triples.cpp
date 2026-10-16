#include "localized_cc/basis_projection.hpp"
#include "localized_cc/canonical_triples.hpp"
#include "localized_cc/laplace_triples.hpp"
#include "localized_cc/parallel.hpp"
#include "localized_cc/partition.hpp"
#include "localized_cc/selected_ccsd_manifold.hpp"
#include "localized_cc/selected_ccsd_storage.hpp"
#include "localized_cc/semicanonical_triples.hpp"

#include "modernqc/linalg/matrix.hpp"
#include "../src/triples_block_screen.hpp"
#include "../src/triples_monomer_patterns.hpp"
#include "../src/hermitian_one_body.hpp"
#include "../src/triples_row_file.hpp"
#include "../src/triples_occupied_panel.hpp"
#include "../src/triples_virtual_axes.hpp"
#include "../src/triples_sector_checkpoint.hpp"
#include <cstdlib>
#include "modernqc/linalg/eigensolver.hpp"
#include <limits>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using localized_cc::Matrix;
using localized_cc::SpatialRhfHamiltonian;
using localized_cc::SymmetricEri4;
using localized_cc::Tensor4;
using ModernMatrix = modernqc::linalg::Matrix;

void require(const bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void close(const double actual, const double expected, const double tolerance,
           const char* message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > tolerance)
    throw std::runtime_error(message);
}

#if defined(__linux__)
// A unique directory per MPI rank; never remove a user-supplied tree.
struct TemporaryTriplesDirectory {
  std::filesystem::path path;
  TemporaryTriplesDirectory() {
    const auto pattern=(std::filesystem::temp_directory_path()/"triples-test-XXXXXX").string();
    std::vector<char> name(pattern.begin(),pattern.end()); name.push_back('\0');
    if(!::mkdtemp(name.data())) throw std::runtime_error("cannot create triples test scratch directory");
    path=name.data();
  }
  ~TemporaryTriplesDirectory() {
    std::error_code error;
    std::filesystem::remove(path,error); // empty directory only
  }
};

void test_triples_row_file() {
  TemporaryTriplesDirectory directory;
  {
    using localized_cc::detail::TriplesRowFile;
    const auto path=(directory.path/"durable.bin").string();
    {
      TriplesRowFile file(directory.path.string(),3,2,"durable.bin","fixture");
      file.set(0,0,1.25); file.set(0,1,-2.5); file.commit_row(0);
      file.set(1,0,99.); // interrupted, uncommitted partial row
      bool rejected=false;
      try { TriplesRowFile other(directory.path.string(),3,2,"durable.bin","fixture"); }
      catch(const std::runtime_error&) { rejected=true; }
      require(rejected,"concurrent moment writer must be rejected");
    }
    {
      TriplesRowFile file(directory.path.string(),3,2,"durable.bin","fixture");
      require(file.completed_rows()==1,"partial row incorrectly committed");
      file.set(1,0,3.); file.set(1,1,4.); file.commit_row(1);
      file.commit_row(2); // valid all-zero row
      file.finish_writes();
      std::vector<double> values(6); file.read(0,values);
      require(values==std::vector<double>({1.25,-2.5,3.,4.,0.,0.}),"moment row restart differs");
    }
    bool rejected=false;
    try { TriplesRowFile file(directory.path.string(),3,2,"durable.bin","wrong"); }
    catch(const std::runtime_error&) { rejected=true; }
    require(rejected,"moment identity mismatch accepted");
    {
      std::fstream corrupt(path,std::ios::in|std::ios::out|std::ios::binary);
      const double changed=42.; corrupt.write(reinterpret_cast<const char*>(&changed),sizeof(changed));
    }
    rejected=false;
    try { TriplesRowFile file(directory.path.string(),3,2,"durable.bin","fixture"); }
    catch(const std::runtime_error&) { rejected=true; }
    require(rejected,"corrupted committed moment row accepted");
    std::filesystem::remove(path); std::filesystem::remove(path+".rows");
  }
  {
    localized_cc::detail::TriplesRowFile file(directory.path.string(),3,2);
    std::vector<double> row(2);
    bool rejected=false;
    try { file.read(0,row); } catch(const std::out_of_range&) { rejected=true; }
    require(rejected,"scratch read before sealing must fail");
    file.set(0,0,1.25); file.set(0,1,-2.5);
    file.set(2,0,3.75); // skipped row and column must stay zero
    rejected=false;
    try { file.set(1,0,7.0); } catch(const std::logic_error&) { rejected=true; }
    require(rejected,"scratch producer must be row-monotone");
    file.finish_writes(); file.finish_writes();
    std::vector<double> rows(6);
    file.read(0,rows);
    require(rows==std::vector<double>({1.25,-2.5,0.,0.,3.75,0.}),"scratch exact row round trip");
    file.read(2,row);
    require(row[0]==3.75 && row[1]==0.,"scratch positional row read");
    rejected=false;
    try { file.read(3,row); } catch(const std::out_of_range&) { rejected=true; }
    require(rejected,"scratch out-of-range read must fail");
    rejected=false;
    try { file.set(2,1,8.0); } catch(const std::out_of_range&) { rejected=true; }
    require(rejected,"sealed scratch writes must fail");
  }
  require(std::filesystem::is_empty(directory.path),"scratch owner cleanup");
  bool rejected=false;
  try { localized_cc::detail::TriplesRowFile missing((directory.path/"missing").string(),1,1); }
  catch(const std::invalid_argument&) { rejected=true; }
  require(rejected,"nonexistent scratch directory must fail");
}
#endif

Tensor4 rotate_eri(const Tensor4& source, const ModernMatrix& rotation) {
  const auto n = rotation.rows();
  Tensor4 result({n, n, n, n});
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s)
          for (std::size_t u = 0; u < n; ++u)
            for (std::size_t v = 0; v < n; ++v)
              for (std::size_t w = 0; w < n; ++w)
                for (std::size_t x = 0; x < n; ++x)
                  result(p, q, r, s) +=
                      rotation(u, p) * rotation(v, q) * rotation(w, r) *
                      rotation(x, s) * source(u, v, w, x);
  return result;
}

Matrix rotate_one_body(const Matrix& source, const ModernMatrix& rotation) {
  const auto n = rotation.rows();
  Matrix result({n, n});
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t u = 0; u < n; ++u)
        for (std::size_t v = 0; v < n; ++v)
          result(p, q) +=
              rotation(u, p) * source(u, v) * rotation(v, q);
  return result;
}

void set_antisymmetric(Tensor4& tensor, const std::size_t i,
                       const std::size_t j, const std::size_t a,
                       const std::size_t b, const double value) {
  tensor(i, j, a, b) = value;
  tensor(j, i, a, b) = -value;
  tensor(i, j, b, a) = -value;
  tensor(j, i, b, a) = value;
}

void test_legacy_one_body_rebuild() {
  // Large one-body entries expose an asymmetry above 1e-12 while still
  // inside the independently computed two-DGEMM rounding bound.
  ModernMatrix ao(2,2), c(2,2);
  ao(0,0)=-10000.; ao(1,1)=10000.;
  ao(0,1)=ao(1,0)=2000.;
  c(0,0)=c(1,1)=1.;
  auto projection=localized_cc::detail::project_hermitian_one_body(ao,c);
  close(projection.value(0,1),2000.,0.,"one-body analytic projection");
  SpatialRhfHamiltonian h;
  h.hcore=projection.value;
  h.fock=projection.value;
  h.fock(0,0)+=2.; h.fock(1,1)+=3.;
  const auto reference=h;
  h.hcore(0,1)+=1e-11;
  h.fock(0,1)+=1e-11;
  ModernMatrix legacy(2,2);
  for (std::size_t p=0;p<2;++p) for (std::size_t q=0;q<2;++q)
    legacy(p,q)=h.fock(p,q);
  bool rejected=false;
  try { (void)modernqc::linalg::diagonalize_symmetric(legacy); }
  catch (const std::invalid_argument&) { rejected=true; }
  require(rejected,"fixture must reproduce old eigensolver failure");
  const auto audit=localized_cc::detail::rebuild_legacy_one_body(h,ao,c);
  require(audit.original_asymmetry>1e-12,"legacy asymmetry recorded");
  for (std::size_t p=0;p<2;++p) for (std::size_t q=0;q<2;++q) {
    close(h.fock(p,q),reference.fock(p,q),0.,"rebuild preserves analytic Fock");
    legacy(p,q)=h.fock(p,q);
  }
  (void)modernqc::linalg::diagonalize_symmetric(legacy); // same default 1e-12
  const auto expect_rejection=[&](SpatialRhfHamiltonian bad) {
    const auto before=bad.fock.data();
    bool failed=false;
    try { (void)localized_cc::detail::rebuild_legacy_one_body(bad,ao,c); }
    catch(const std::runtime_error&) {failed=true;}
    require(failed,"invalid legacy matrix must fail closed");
    require(bad.fock.data()==before,"failed rebuild must not mutate Fock");
  };
  auto bad=reference;
  bad.hcore(0,1)+=1e-3; bad.fock(0,1)+=1e-3;
  expect_rejection(bad); // unrelated hcore corruption is not roundoff
  bad=reference; bad.fock(0,1)+=1e-8;
  expect_rejection(bad); // inconsistent two-electron remainder
  bad=reference; bad.hcore(0,0)=std::numeric_limits<double>::infinity();
  expect_rejection(bad);
  auto bad_ao=ao; bad_ao(0,1)+=1e-8;
  rejected=false;
  try { (void)localized_cc::detail::project_hermitian_one_body(bad_ao,c); }
  catch(const std::runtime_error&) {rejected=true;}
  require(rejected,"AO symmetry tolerance must not be relaxed");
}

void test_laplace_resolvent_and_rotation_invariance() {
  constexpr std::size_t occupied = 2;
  constexpr std::size_t virtuals = 2;
  constexpr std::size_t orbitals = occupied + virtuals;
  Matrix fock({orbitals, orbitals});
  fock(0, 0) = -1.2;
  fock(1, 1) = -0.8;
  fock(2, 2) = 0.4;
  fock(3, 3) = 0.9;
  Tensor4 eri({orbitals, orbitals, orbitals, orbitals});
  for (std::size_t p = 0; p < orbitals; ++p)
    for (std::size_t q = 0; q < orbitals; ++q)
      for (std::size_t r = 0; r < orbitals; ++r)
        for (std::size_t s = 0; s < orbitals; ++s) {
          const auto pq = std::minmax(p, q);
          const auto rs = std::minmax(r, s);
          const auto first = pq < rs ? pq : rs;
          const auto second = pq < rs ? rs : pq;
          eri(p, q, r, s) = 1.0e-3 * std::sin(static_cast<double>(
              1 + 17 * first.first + 11 * first.second +
              7 * second.first + 3 * second.second));
        }
  SpatialRhfHamiltonian canonical{fock, fock, SymmetricEri4(eri), occupied, virtuals, 0,
                                  -2.0, 0.0, 0.0, {}, {0, 1}, {2, 3}};
  Matrix t1({2 * occupied, 2 * virtuals});
  t1(0, 0) = 0.01;
  t1(1, 1) = 0.01;
  t1(2, 2) = -0.008;
  t1(3, 3) = -0.008;
  Tensor4 t2({2 * occupied, 2 * occupied, 2 * virtuals, 2 * virtuals});
  set_antisymmetric(t2, 0, 1, 0, 1, 0.020);
  set_antisymmetric(t2, 0, 2, 0, 2, -0.015);
  set_antisymmetric(t2, 1, 3, 1, 3, -0.015);
  set_antisymmetric(t2, 2, 3, 2, 3, 0.011);

  localized_cc::LaplaceTriplesOptions options;
  options.quadrature_points = 7;
  options.requested_energy_error_hartree = 1.0e-9;
  const auto canonical_laplace =
      localized_cc::solve_localized_triples_laplace_dense_oracle(
          canonical, t1, t2, options);

  ModernMatrix occupied_rotation(occupied, occupied);
  const double co = std::cos(0.37), so = std::sin(0.37);
  occupied_rotation(0, 0) = co;
  occupied_rotation(0, 1) = -so;
  occupied_rotation(1, 0) = so;
  occupied_rotation(1, 1) = co;
  ModernMatrix virtual_rotation(virtuals, virtuals);
  const double cv = std::cos(-0.29), sv = std::sin(-0.29);
  virtual_rotation(0, 0) = cv;
  virtual_rotation(0, 1) = -sv;
  virtual_rotation(1, 0) = sv;
  virtual_rotation(1, 1) = cv;
  ModernMatrix rotation(orbitals, orbitals);
  for (std::size_t p = 0; p < occupied; ++p)
    for (std::size_t q = 0; q < occupied; ++q)
      rotation(p, q) = occupied_rotation(p, q);
  for (std::size_t p = 0; p < virtuals; ++p)
    for (std::size_t q = 0; q < virtuals; ++q)
      rotation(occupied + p, occupied + q) = virtual_rotation(p, q);
  SpatialRhfHamiltonian localized{rotate_one_body(fock, rotation),
                                  rotate_one_body(fock, rotation),
                                  SymmetricEri4(rotate_eri(eri, rotation)),
                                  occupied, virtuals, 0, -2.0, 0.0, 0.0,
                                  {}, {0, 1}, {2, 3}};
  const auto localized_amplitudes = localized_cc::project_ccsd_amplitudes(
      t1, t2, modernqc::linalg::transpose(occupied_rotation),
      modernqc::linalg::transpose(virtual_rotation));
  const auto localized_laplace =
      localized_cc::solve_localized_triples_laplace_dense_oracle(
          localized, localized_amplitudes.t1, localized_amplitudes.t2,
          options);
  localized_cc::StreamedLaplaceTriplesOptions streamed_options;
  streamed_options.quadrature_points = options.quadrature_points;
  // This known case validates the streamed contractions against the dense
  // oracle. The separately reported certified bound is tested below; it is
  // deliberately not used to reject this small algebraic equivalence case.
  streamed_options.requested_energy_error_hartree = 1.0;
  streamed_options.virtual_tile_size = 2;
  streamed_options.moment_backend =
      localized_cc::CanonicalTriplesBackend::tiled_cpu;
  const auto streamed =
      localized_cc::solve_localized_triples_laplace_streamed(
          localized, localized_amplitudes.t1, localized_amplitudes.t2,
          std::vector<int>{0, 1}, std::vector<int>{0, 1},
          streamed_options);
  {
    auto batched = streamed_options;
    batched.propagation_row_batch = 1;
    const auto result = localized_cc::solve_localized_triples_laplace_streamed(
        localized, localized_amplitudes.t1, localized_amplitudes.t2,
        std::vector<int>{0, 1}, std::vector<int>{0, 1}, batched);
    close(result.parenthetical_energy, streamed.parenthetical_energy, 1.e-13, "bounded propagation panels");
    require(result.many_body_terms == streamed.many_body_terms, "bounded panel class counts");
    for (std::size_t m = 0; m < 6; ++m)
      close(result.many_body_energies[m], streamed.many_body_energies[m], 1.e-13, "bounded panel nM energies");
    batched.initial_point_many_body_energies = result.point_many_body_energies;
    batched.completed_points = result.completed_points;
    const auto restarted = localized_cc::solve_localized_triples_laplace_streamed(
        localized, localized_amplitudes.t1, localized_amplitudes.t2,
        std::vector<int>{0, 1}, std::vector<int>{0, 1}, batched);
    close(restarted.parenthetical_energy, result.parenthetical_energy, 1.e-13, "completed point restart");
  }
  {
    auto limited = streamed_options;
    limited.maximum_additional_memory_bytes = 1;
    bool allocation_started = false;
    limited.progress_observer = [&](const std::string& message) {
      if (message.find("allocating two persistent") != std::string::npos) allocation_started = true;
    };
    bool rejected = false;
    try {
      (void)localized_cc::solve_localized_triples_laplace_streamed(
          localized, localized_amplitudes.t1, localized_amplitudes.t2,
          std::vector<int>{0, 1}, std::vector<int>{0, 1}, limited);
    } catch (const std::runtime_error&) { rejected = true; }
    require(rejected && !allocation_started, "memory guard must reject before moment allocation");
  }
  // Independently compare each cheap bound with actual constructed moments.
  // This catches permutation/prefactor/index errors even when a conservative
  // cumulative budget happens to screen no molecular blocks.
  using BlockKey=std::array<std::size_t,6>;
  std::map<BlockKey,std::pair<double,double>> block_bounds;
  (void)localized_cc::detail::plan_triples_block_screen(
      localized,localized_amplitudes.t1,localized_amplitudes.t2,2,1.0,1.e-7,
      std::numeric_limits<std::size_t>::max(),
      [&](const auto& occ,const auto& offsets,double m,double l) {
        block_bounds[{occ[0],occ[1],occ[2],offsets[0],offsets[1],offsets[2]}]={m,l};
      });
  localized_cc::LocalizedTriplesMomentStreamOptions moment_check;
  moment_check.backend=localized_cc::CanonicalTriplesBackend::tiled_cpu;
  moment_check.virtual_tile_size=2;
  moment_check.observer=[&](localized_cc::LocalizedTriplesMomentBlock&& block) {
    const auto& o=block.occupied_indices;
    const auto& a=block.virtual_offsets;
    const auto it=block_bounds.find({o[0],o[1],o[2],a[0],a[1],a[2]});
    double mn=0,ln=0;
    for (auto x:block.doubles_moment) mn=std::hypot(mn,x);
    for (auto x:block.left_moment) ln=std::hypot(ln,x);
    const auto limit=it==block_bounds.end() ? std::pair{0.0,0.0} : it->second;
    require(mn<=limit.first+1.e-14 && ln<=limit.second+1.e-14,
            "preconstruction block norm underestimates actual triples moments");
  };
  localized_cc::stream_localized_triples_moments(
      localized,localized_amplitudes.t1,localized_amplitudes.t2,moment_check);
  localized_cc::SelectedCcsdManifoldOptions selected_options;
  selected_options.maximum_monomers = 2;
  const auto selected_plan = localized_cc::plan_selected_ccsd_manifold(
      std::vector<int>{0, 0, 1, 1}, std::vector<int>{0, 0, 1, 1},
      selected_options);
  const localized_cc::SelectedDoublesLayout selected_layout(selected_plan);
  const auto selected_blocks =
      selected_layout.pack(localized_amplitudes.t2);
  const localized_cc::SelectedDoublesView selected_view(selected_layout,
                                                         selected_blocks);
  const auto streamed_selected =
      localized_cc::solve_localized_triples_laplace_streamed(
          localized, localized_amplitudes.t1, selected_view,
          std::vector<int>{0, 1}, std::vector<int>{0, 1},
          streamed_options);
  // Preconstruction bounds must survive NONDIAGONAL propagation. In
  // particular, tiny omitted source moments can feed other external nM
  // classes. These checks do not replace the molecular timing/energy gate.
  for (const double budget : {0.0, 1.e-12, 1.e-7, 1.0}) {
    auto screened_options=streamed_options;
    screened_options.block_screening_budget_hartree=budget;
    const auto screened=localized_cc::solve_localized_triples_laplace_streamed(
        localized,localized_amplitudes.t1,localized_amplitudes.t2,
        std::vector<int>{0,1},std::vector<int>{0,1},screened_options);
    require(screened.block_screening_bound_hartree<=budget,
            "screen exceeded its cumulative energy budget");
    const double gate=screened.block_screening_bound_hartree+1.e-13;
    close(screened.parenthetical_energy,streamed.parenthetical_energy,gate,
          "screen omitted more total energy than its bound");
    for (std::size_t n=0;n<6;++n)
      close(screened.many_body_energies[n],streamed.many_body_energies[n],gate,
            "screen omitted more nM energy than its bound");
    require(screened.many_body_terms==streamed.many_body_terms,
            "screen changed the full nM manifold");
    close(screened.partition_error,0.0,1.e-13,"screen partition closure");
    if (budget==0.0)
      require(screened.block_screening_candidates==0,
              "disabled screen performed planning work");
    if (budget==1.0) {
      require(screened.block_screening_skipped>0 &&
              screened.block_screening_skipped==screened.block_screening_candidates,
              "synthetic complete-screen gate did not omit all moment blocks");
      close(screened.parenthetical_energy,0.0,0.0,
            "complete screen did not bypass moment construction");
    }
  }
  {
    auto bad=streamed_options;
    bad.block_screening_budget_hartree=-1.0;
    bool rejected=false;
    try {
      (void)localized_cc::solve_localized_triples_laplace_streamed(
          localized,localized_amplitudes.t1,localized_amplitudes.t2,
          std::vector<int>{0,1},std::vector<int>{0,1},bad);
    } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"negative screen budget accepted");
    bad.block_screening_budget_hartree=1.e-7;
    bad.completed_points.assign(7,0U);
    rejected=false;
    try {
      (void)localized_cc::solve_localized_triples_laplace_streamed(
          localized,localized_amplitudes.t1,localized_amplitudes.t2,
          std::vector<int>{0,1},std::vector<int>{0,1},bad);
    } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"screen accepted a point restart without mask provenance");
  }
  localized_cc::CanonicalTriplesOptions reference_options;
  const auto semicanonical =
      localized_cc::solve_localized_triples_semicanonical(
          localized, localized_amplitudes.t1, localized_amplitudes.t2,
          reference_options);

  close(canonical_laplace.parenthetical_energy,
        localized_laplace.parenthetical_energy, 5.0e-13,
        "Laplace triples occupied/virtual rotation invariance");
  close(localized_laplace.exact_resolvent_energy,
        semicanonical.triples.parenthetical_energy, 1.0e-12,
        "Laplace exact resolvent/semicanonical CCSD(T) agreement");
  close(streamed.parenthetical_energy,
        localized_laplace.parenthetical_energy, 1.0e-12,
        "streamed/dense orbital-invariant Laplace triples agreement");
  close(streamed_selected.parenthetical_energy,
        streamed.parenthetical_energy, 1.0e-13,
        "selected-block/dense T2 Laplace triples agreement");
  for (std::size_t order = 0; order < streamed.many_body_energies.size();
       ++order)
    close(streamed_selected.many_body_energies[order],
          streamed.many_body_energies[order], 1.0e-13,
          "selected-block triples many-body partition agreement");
  close(streamed.many_body_energies[0] + streamed.many_body_energies[1],
        streamed.parenthetical_energy, 1.0e-13,
        "W2 streamed triples M1+M2 partition closure");
  for (std::size_t order = 2; order < streamed.many_body_energies.size();
       ++order)
    close(streamed.many_body_energies[order], 0.0, 0.0,
          "W2 streamed triples cannot contain M3-M6 terms");
  const auto streamed_partition =
      localized_cc::partition_triples_many_body_sums(
          streamed.parenthetical_energy, streamed.many_body_energies,
          streamed.many_body_terms);
  close(streamed_partition.sum_error, 0.0, 1.0e-13,
        "streamed triples public partition closure");
  require(std::abs(localized_laplace.quadrature_energy_error) <=
              localized_laplace.certified_energy_error_bound,
          "Laplace triples certified error bound");
  require(localized_laplace.localized_fock_offdiagonal > 1.0e-3,
          "Laplace test did not exercise noncanonical Fock coupling");

  localized_cc::LaplaceTriplesOptions high_order_options = options;
  high_order_options.quadrature_scheme =
      localized_cc::LaplaceTriplesQuadratureScheme::
          scaled_gauss_laguerre;
  high_order_options.quadrature_points = 64;
  const auto high_order_dense =
      localized_cc::solve_localized_triples_laplace_dense_oracle(
          localized, localized_amplitudes.t1, localized_amplitudes.t2,
          high_order_options);
  localized_cc::StreamedLaplaceTriplesOptions high_order_streamed_options;
  high_order_streamed_options.quadrature_scheme =
      high_order_options.quadrature_scheme;
  high_order_streamed_options.quadrature_points =
      high_order_options.quadrature_points;
  high_order_streamed_options.requested_energy_error_hartree = 1.0e-9;
  high_order_streamed_options.virtual_tile_size = 2;
  high_order_streamed_options.moment_backend =
      localized_cc::CanonicalTriplesBackend::tiled_cpu;
  const auto high_order_streamed =
      localized_cc::solve_localized_triples_laplace_streamed(
          localized, localized_amplitudes.t1, localized_amplitudes.t2,
          std::vector<int>{0, 1}, std::vector<int>{0, 1},
          high_order_streamed_options);
  close(high_order_streamed.parenthetical_energy,
        high_order_dense.parenthetical_energy, 1.0e-12,
        "scaled Gauss-Laguerre streamed/dense agreement");
  require(high_order_dense.certified_energy_error_bound <= 1.0e-9,
          "scaled Gauss-Laguerre dense certified bound");
  require(high_order_streamed.certified_quadrature_error_bound <= 1.0e-9,
          "scaled Gauss-Laguerre streamed certified bound");
  require(high_order_dense.maximum_reciprocal_error <
              localized_laplace.maximum_reciprocal_error,
          "scaled Gauss-Laguerre should improve reciprocal accuracy");

  bool rejected_invalid_minimax_order = false;
  try {
    auto invalid_options = options;
    invalid_options.quadrature_points = 10;
    (void)localized_cc::solve_localized_triples_laplace_dense_oracle(
        localized, localized_amplitudes.t1, localized_amplitudes.t2,
        invalid_options);
  } catch (const std::invalid_argument& error) {
    rejected_invalid_minimax_order =
        std::string(error.what()).find("supports 5, 6, or 7 points") !=
        std::string::npos;
  }
  require(rejected_invalid_minimax_order,
          "invalid q10 minimax order should fail with a specific diagnostic");
}

void test_shared_moment_packing() {
  constexpr std::size_t o=6,v=11,n=o+v;
  Matrix f({n,n}), t1({o,v});
  Tensor4 eri({n,n,n,n}), t2({o,o,v,v});
  for (std::size_t p=0;p<n;++p) f(p,p)=p<o ? -1.0 : 0.5;
  for (std::size_t p=0;p<n;++p) for (std::size_t q=0;q<n;++q)
    for (std::size_t r=0;r<n;++r) for (std::size_t s=0;s<n;++s)
      eri(p,q,r,s)=1.e-3*std::sin(double(1+2*p+3*q+5*r+7*s));
  for (std::size_t i=0;i<o;++i) for (std::size_t a=0;a<v;++a)
    t1(i,a)=1.e-3*std::cos(double(1+i+a));
  for (std::size_t i=0;i<o;++i) for (std::size_t j=i+1;j<o;++j)
    for (std::size_t a=0;a<v;++a) for (std::size_t b=a+1;b<v;++b)
      set_antisymmetric(t2,i,j,a,b,0.01*std::sin(double(1+3*i+5*j+7*a+b)));
  localized_cc::LocalizedTriplesMomentStreamOptions opt;
  opt.virtual_tile_size=8; // parallel pack region and ragged final tile
  std::vector<localized_cc::LocalizedTriplesMomentBlock> expected;
  opt.observer=[&](auto&& block) { expected.push_back(std::move(block)); };
  localized_cc::stream_localized_triples_moments(f,eri,t1,t2,opt);
  std::size_t index=0;
  opt.shared_pair_packing=true;
  opt.observer=[&](auto&& block) {
    require(index<expected.size(),"shared pack added a moment block");
    const auto& ref=expected[index++];
    require(block.occupied_indices==ref.occupied_indices &&
            block.virtual_offsets==ref.virtual_offsets &&
            block.virtual_dimensions==ref.virtual_dimensions,"shared pack index mismatch");
    require(block.doubles_moment.size()==ref.doubles_moment.size() &&
            block.left_moment.size()==ref.left_moment.size(),"shared pack dimension mismatch");
    for (std::size_t k=0;k<ref.doubles_moment.size();++k) {
      close(block.doubles_moment[k],ref.doubles_moment[k],1.e-13,"shared pack D");
      close(block.left_moment[k],ref.left_moment[k],1.e-13,"shared pack L");
    }
  };
  localized_cc::stream_localized_triples_moments(f,eri,t1,t2,opt);
  require(index==expected.size(),"shared pack dropped a moment block");
}

void test_screening_six_monomer_labels() {
  constexpr std::size_t o=3,v=3,n=o+v;
  Matrix f({n,n});
  for (std::size_t p=0;p<n;++p) f(p,p)=p<o ? -1.2+0.1*p : 0.3+0.1*(p-o);
  f(0,1)=f(1,0)=0.04;
  f(3,4)=f(4,3)=-0.03;
  Tensor4 eri({n,n,n,n});
  for (std::size_t p=0;p<n;++p) for (std::size_t q=0;q<n;++q)
    for (std::size_t r=0;r<n;++r) for (std::size_t s=0;s<n;++s) {
      std::pair<std::size_t,std::size_t> a=std::minmax(p,q),b=std::minmax(r,s);
      if (b<a) std::swap(a,b);
      eri(p,q,r,s)=1.e-3*std::sin(double(1+17*a.first+11*a.second+7*b.first+3*b.second));
    }
  SpatialRhfHamiltonian h{f,f,SymmetricEri4(eri),o,v,0,-3.,0.,0.,{},{0,1,2},{3,4,5}};
  Matrix t1({2*o,2*v});
  Tensor4 t2({2*o,2*o,2*v,2*v});
  for (std::size_t i=0;i<2*o;++i) for (std::size_t a=0;a<2*v;++a)
    if (i%2==a%2) t1(i,a)=1.e-3*std::cos(double(1+i+a));
  for (std::size_t i=0;i<2*o;++i) for (std::size_t j=i+1;j<2*o;++j)
    for (std::size_t a=0;a<2*v;++a) for (std::size_t b=a+1;b<2*v;++b)
      if ((i%2+j%2)==(a%2+b%2))
        set_antisymmetric(t2,i,j,a,b,0.01*std::sin(double(1+3*i+5*j+7*a+b)));
  // Closed-shell spin-flip invariant fixture, without modifying production data.
  const auto raw_t1=t1;
  const auto raw_t2=t2;
  for(std::size_t i=0;i<2*o;++i) for(std::size_t a=0;a<2*v;++a)
    t1(i,a)=0.5*(raw_t1(i,a)+raw_t1(i^1U,a^1U));
  for(std::size_t i=0;i<2*o;++i) for(std::size_t j=0;j<2*o;++j)
    for(std::size_t a=0;a<2*v;++a) for(std::size_t b=0;b<2*v;++b)
      t2(i,j,a,b)=0.5*(raw_t2(i,j,a,b)+raw_t2(i^1U,j^1U,a^1U,b^1U));
  localized_cc::StreamedLaplaceTriplesOptions opt;
  opt.moment_backend=localized_cc::CanonicalTriplesBackend::tiled_cpu;
  opt.virtual_tile_size=2;
  opt.requested_energy_error_hartree=1.e-6;
  const auto ref=localized_cc::solve_localized_triples_laplace_streamed(
      h,t1,t2,{0,1,2},{3,4,5},opt);
  require(ref.many_body_terms[5]>0,"six-label oracle does not contain M6");
  require(std::abs(ref.many_body_energies[5])>0.0,"six-label oracle has zero M6 energy");
  {
    auto reduced=opt; reduced.sector_streaming=true; reduced.two_spin_sectors=true;
    const auto actual=localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},{3,4,5},reduced);
    close(actual.parenthetical_energy,ref.parenthetical_energy,1e-13,"two-sector total");
    close(actual.certified_quadrature_error_bound,ref.certified_quadrature_error_bound,1e-15,"two-sector bound");
    require(actual.many_body_terms==ref.many_body_terms,"two-sector full term counts");
    for(std::size_t m=0;m<6;++m)
      close(actual.many_body_energies[m],ref.many_body_energies[m],1e-13,"two-sector M1-M6");
    for(const std::vector<int> labels : {std::vector<int>{0,0,0},std::vector<int>{0,0,1},std::vector<int>{0,1,2}}) {
      const auto full=localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,labels,labels,opt);
      const auto pair=localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,labels,labels,reduced);
      require(pair.many_body_terms==full.many_body_terms,"two-sector repeated-label counts");
      for(std::size_t m=0;m<6;++m)
        close(pair.many_body_energies[m],full.many_body_energies[m],1e-13,"two-sector repeated-label nM");
    }
    bool rejected=false;
    auto broken=t1; broken(0,0)+=1e-12;
    try { (void)localized_cc::solve_localized_triples_laplace_streamed(h,broken,t2,{0,1,2},{3,4,5},reduced); }
    catch(const std::runtime_error&) { rejected=true; }
    require(rejected,"two-sector gate accepted spin-asymmetric amplitudes");
    if(localized_cc::parallel::world().rank==0)
      std::cout << "Two-sector closed-shell total/nM/bound and spin-rejection gates passed\n";
  }
  // Exact moment-level comparison before the Laplace propagation: catches
  // index, sign, multiplicity and stride errors that energy cancellation hides.
  using Key = std::array<std::size_t,6>;
  using Moments = std::map<Key,std::pair<double,double>>;
  const auto collect = [&](const bool spin_aware, const std::size_t width,
                           const bool shared_pairs) {
    Moments values;
    localized_cc::LocalizedTriplesMomentStreamOptions stream;
    stream.virtual_tile_size=width;
    stream.shared_pair_packing=shared_pairs;
    stream.spin_aware_construction=spin_aware;
    stream.observer=[&](auto&& block) {
      const auto ijk=block.occupied_indices;
      const auto dims=block.virtual_dimensions;
      for(std::size_t x=0;x<dims[0];++x)
        for(std::size_t y=0;y<dims[1];++y)
          for(std::size_t z=0;z<dims[2];++z) {
            const auto a=block.virtual_offsets[0]+block.virtual_stride*x;
            const auto b=block.virtual_offsets[1]+block.virtual_stride*y;
            const auto c=block.virtual_offsets[2]+block.virtual_stride*z;
            if(!(a<b && b<c)) continue;
            const bool allowed=(a%2+b%2+c%2)==(ijk[0]%2+ijk[1]%2+ijk[2]%2);
            if(spin_aware) require(allowed,"spin-aware emitted forbidden spin tuple");
            if(!allowed) continue;
            const auto k=(x*dims[1]+y)*dims[2]+z;
            require(values.emplace(Key{ijk[0],ijk[1],ijk[2],a,b,c},
                std::make_pair(block.doubles_moment[k],block.left_moment[k])).second,
                "duplicate moment tuple");
          }
    };
    localized_cc::stream_localized_triples_moments(h,t1,t2,stream);
    return values;
  };
  const auto oracle_moments=collect(false,2,false);
  // Occupied triples have cyclic MPI ownership: an individual rank need not
  // own all four sectors (the same-spin sectors have only one row each here).
  // Assert global coverage, then retain the rank-local tuple-by-tuple oracle.
  std::vector<double> sector_coverage(4,0.0);
  for(const auto& [key,value]:oracle_moments) {
    (void)value;
    ++sector_coverage[key[0]%2+key[1]%2+key[2]%2];
  }
  localized_cc::parallel::sum_in_place(sector_coverage);
  // Three alpha and three beta orbitals yield 1,9,9,1 occupied triples;
  // each has the same number of allowed virtual triples: 1,81,81,1 tuples.
  const std::array<double,4> expected_coverage{1.0,81.0,81.0,1.0};
  for(std::size_t sector=0;sector<4;++sector)
    require(sector_coverage[sector]==expected_coverage[sector],
            "exact-channel oracle global spin-sector tuple count mismatch");
  for(const std::size_t width:{1U,2U,3U,5U,8U}) for(const bool shared:{false,true}) {
    const auto actual=collect(true,width,shared);
    require(actual.size()==oracle_moments.size(),"spin-aware moment coverage");
    for(const auto& [key,value]:oracle_moments) {
      const auto found=actual.find(key);
      require(found!=actual.end(),"missing spin-allowed tuple");
      close(found->second.first,value.first,1.e-13,"spin-aware D moment");
      close(found->second.second,value.second,1.e-13,"spin-aware L moment");
    }
  }
  if(localized_cc::parallel::world().rank==0)
    std::cout << "Virtual exchange DGEMM reuse tuple-level D/L oracle passed; widths=1,2,3,5,8; all spin sectors\n";
  for(const std::size_t width:{2U,5U}) for(const bool sector_streaming:{false,true}) {
    auto aware=opt;
    aware.spin_aware_construction=true;
    aware.shared_pair_packing=true;
    aware.virtual_tile_size=width;
    aware.sector_streaming=sector_streaming;
    const auto result=localized_cc::solve_localized_triples_laplace_streamed(
        h,t1,t2,{0,1,2},{3,4,5},aware);
    close(result.parenthetical_energy,ref.parenthetical_energy,1.e-13,"spin-aware full (T)");
    close(result.certified_quadrature_error_bound,ref.certified_quadrature_error_bound,
          1.e-15,"spin-aware quadrature certification");
    require(result.many_body_terms==ref.many_body_terms,"spin-aware nM term counts");
    for(std::size_t m=0;m<6;++m)
      close(result.many_body_energies[m],ref.many_body_energies[m],1.e-13,"spin-aware M1-M6");
    aware.initial_point_many_body_energies=ref.point_many_body_energies;
    aware.completed_points=ref.completed_points;
    aware.completed_points[1]=0;
    const auto resumed=localized_cc::solve_localized_triples_laplace_streamed(
        h,t1,t2,{0,1,2},{3,4,5},aware);
    require(resumed.restarted_points==opt.quadrature_points-1,"spin-aware restart count");
    close(resumed.parenthetical_energy,ref.parenthetical_energy,1.e-13,"spin-aware restarted total");
    for(std::size_t m=0;m<6;++m)
      close(resumed.many_body_energies[m],ref.many_body_energies[m],1.e-13,"spin-aware restarted nM");
  }
  // The first three occupied and virtual monomers can coincide: cover a
  // nonzero M1 case as well as the disjoint six-label fixture above.
  for(const std::vector<int> labels : {std::vector<int>{0,0,0},std::vector<int>{0,1,2}}) {
    auto aware=opt; aware.spin_aware_construction=true; aware.virtual_tile_size=5;
    const auto baseline=localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,labels,labels,opt);
    const auto result=localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,labels,labels,aware);
    require(result.many_body_terms==baseline.many_body_terms,"spin-aware repeated-label coverage");
    for(std::size_t m=0;m<6;++m)
      close(result.many_body_energies[m],baseline.many_body_energies[m],1.e-13,"spin-aware repeated-label nM");
  }
  {
    auto invalid=opt; invalid.spin_aware_construction=true;
    invalid.moment_backend=localized_cc::CanonicalTriplesBackend::cuda;
    bool rejected=false;
    try { (void)localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},{3,4,5},invalid); }
    catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"spin-aware CUDA must fail closed");
  }
  if(localized_cc::parallel::world().rank==0)
    std::cout << "Spin-aware triples moment/total/nM/restart gates passed\n";
  for (const auto width : {2U,5U}) {
    auto shared=opt;
    shared.shared_pair_packing=true;
    shared.virtual_tile_size=width;
    const auto actual=localized_cc::solve_localized_triples_laplace_streamed(
        h,t1,t2,{0,1,2},{3,4,5},shared);
    close(actual.parenthetical_energy,ref.parenthetical_energy,1.e-13,"shared full (T)");
    require(actual.many_body_terms==ref.many_body_terms,"shared pack nM term coverage");
    for (std::size_t m=0;m<6;++m)
      close(actual.many_body_energies[m],ref.many_body_energies[m],1.e-13,"shared M1-M6");
  }
  {
    auto compact = opt;
    compact.sector_streaming = true;
    compact.shared_pair_packing = true;
    compact.propagation_row_batch = 1; // multiple MPI panels, not just one DGEMM
    std::size_t finished_sectors = 0;
    compact.progress_observer = [&](const std::string& message) {
      if (message.find("finished all pending points; moments released") != std::string::npos)
        ++finished_sectors;
    };
    std::size_t callbacks = 0;
    compact.checkpoint_observer = [&](const auto& state) {
      // Progress is rank-zero-only; the checkpoint callback is collective.
      if (localized_cc::parallel::world().rank == 0)
        require(finished_sectors == 4, "partial sector must never be a completed point");
      ++callbacks;
      require(state.point_many_body_energies.size() == 6 * opt.quadrature_points,
              "compact checkpoint six-class layout");
    };
    const auto low = localized_cc::solve_localized_triples_laplace_streamed(
        h,t1,t2,{0,1,2},{3,4,5},compact);
    require(callbacks == opt.quadrature_points, "compact completed point callbacks");
    require(low.many_body_terms == ref.many_body_terms, "compact lost spin sectors or nM terms");
    require(low.maximum_local_moment_bytes <= ref.maximum_local_moment_bytes,
            "compact moment storage cannot exceed all-sector storage");
    close(low.parenthetical_energy,ref.parenthetical_energy,1.e-13,"compact full energy");
    close(low.partition_error,0.,1.e-13,"compact independent closure");
    close(low.certified_quadrature_error_bound,ref.certified_quadrature_error_bound,
          1.e-15,"compact unchanged quadrature bound");
    for (std::size_t n=0;n<6;++n)
      close(low.many_body_energies[n],ref.many_body_energies[n],1.e-13,"compact M1-M6");
#if defined(__linux__)
    {
      TemporaryTriplesDirectory directory;
      auto disk=compact;
      disk.progress_observer={}; disk.checkpoint_observer={};
      disk.scratch_directory=directory.path.string();
      disk.maximum_scratch_bytes=1024*1024;
      for(const bool two_sectors:{false,true}) {
        auto durable=disk;
        durable.two_spin_sectors=two_sectors;
        durable.sector_checkpoint_path=(directory.path/"sector.chk").string();
        durable.sector_checkpoint_identity="six-monomer-fixed-fixture-v1";
        durable.moment_checkpoint_directory=directory.path.string();
        durable.sector_checkpoint_observer=[](std::size_t sector,std::size_t point) {
          if(sector==1 && point==0) throw std::runtime_error("injected triples interruption");
        };
        bool interrupted=false;
        try { (void)localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},{3,4,5},durable); }
        catch(const std::runtime_error& e) { interrupted=std::string(e.what())=="injected triples interruption"; if(!interrupted) throw; }
        require(interrupted,"durable triples interruption not reached");
        durable.sector_checkpoint_observer={};
        std::size_t restored=0;
        durable.progress_observer=[&](const std::string& s) {
          if(s.find("restored; moment construction skipped")!=std::string::npos) ++restored;
        };
        const auto resumed=localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},{3,4,5},durable);
        if(localized_cc::parallel::world().rank==0) require(restored==1,"completed sector not reused");
        close(resumed.parenthetical_energy,ref.parenthetical_energy,1e-13,"durable total");
        close(resumed.partition_error,0.,1e-13,"durable independent total closure");
        close(resumed.certified_quadrature_error_bound,ref.certified_quadrature_error_bound,1e-15,"durable bound");
        require(resumed.many_body_terms==ref.many_body_terms,"durable nM coverage");
        for(std::size_t m=0;m<6;++m) close(resumed.many_body_energies[m],ref.many_body_energies[m],1e-13,"durable nM");
        const auto replay=localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},{3,4,5},durable);
        close(replay.parenthetical_energy,ref.parenthetical_energy,1e-13,"durable fully completed replay");
        auto wrong_mode=durable; wrong_mode.two_spin_sectors=!two_sectors;
        bool mode_rejected=false;
        try { (void)localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},{3,4,5},wrong_mode); }
        catch(const std::runtime_error&) { mode_rejected=true; }
        require(mode_rejected,"checkpoint accepted different spin-sector multiplicity");
        auto invalid=durable; invalid.sector_checkpoint_identity+="changed";
        bool rejected=false;
        try { (void)localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},{3,4,5},invalid); }
        catch(const std::runtime_error&) { rejected=true; }
        require(rejected,"durable wrong identity accepted");
        if(localized_cc::parallel::world().rank==0)
          std::filesystem::remove(durable.sector_checkpoint_path);
        for(const auto& entry:std::filesystem::directory_iterator(directory.path))
          if(entry.path().filename().string().starts_with("rank"))
            std::filesystem::remove(entry.path()); // only this test's unique directory
        if(localized_cc::parallel::world().rank==0)
          std::cout << "Durable moment rows and sector/point restart M1-M6 gates passed\n";
      }
      for(const bool parallel_panels:{false,true})
      for(const std::size_t panel_rows:{1U,2U,7U}) {
        disk.propagation_row_batch=panel_rows;
        disk.disk_parallel_panels=parallel_panels;
        disk.spin_aware_construction=(panel_rows==2);
        const auto result=localized_cc::solve_localized_triples_laplace_streamed(
            h,t1,t2,{0,1,2},{3,4,5},disk);
        require(result.many_body_terms==ref.many_body_terms,"disk lost spin sectors/nM terms");
        require(result.maximum_local_moment_bytes==0,"disk mode retained a full moment array");
        require((result.disk_panel_waves>0)==parallel_panels,
                "parallel disk schedule not exercised");
        require(std::isfinite(result.disk_panel_transform_seconds),"disk transform timing");
        require(result.planned_scratch_bytes<=disk.maximum_scratch_bytes,"disk reservation budget");
        close(result.parenthetical_energy,ref.parenthetical_energy,1.e-13,"disk full energy");
        close(result.partition_error,0.,1.e-13,"disk independent closure");
        close(result.certified_quadrature_error_bound,ref.certified_quadrature_error_bound,
              1.e-15,"disk unchanged quadrature bound");
        for(std::size_t n=0;n<6;++n)
          close(result.many_body_energies[n],ref.many_body_energies[n],1.e-13,"disk M1-M6");
        require(std::filesystem::is_empty(directory.path),"solver scratch cleanup");
      }
      {
        auto one_monomer=disk;
        one_monomer.spin_aware_construction=true;
        const auto baseline=localized_cc::solve_localized_triples_laplace_streamed(
            h,t1,t2,{0,0,0},{0,0,0},opt);
        const auto actual=localized_cc::solve_localized_triples_laplace_streamed(
            h,t1,t2,{0,0,0},{0,0,0},one_monomer);
        require(actual.many_body_terms==baseline.many_body_terms,"parallel disk M1 terms");
        for(std::size_t m=0;m<6;++m)
          close(actual.many_body_energies[m],baseline.many_body_energies[m],1.e-13,"parallel disk repeated labels");
      }
      disk.spin_aware_construction=true;
      disk.initial_point_many_body_energies=ref.point_many_body_energies;
      disk.completed_points=ref.completed_points;
      disk.completed_points[1]=0;
      const auto resumed_disk=localized_cc::solve_localized_triples_laplace_streamed(
          h,t1,t2,{0,1,2},{3,4,5},disk);
      require(resumed_disk.restarted_points==opt.quadrature_points-1,"disk partial point restart count");
      for(std::size_t n=0;n<6;++n)
        close(resumed_disk.many_body_energies[n],ref.many_body_energies[n],1.e-13,"disk restart M1-M6");
      disk.initial_point_many_body_energies.clear(); disk.completed_points.clear();
      {
        auto invalid=disk; invalid.scratch_directory.clear();
        bool rejected=false;
        try { (void)localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},{3,4,5},invalid); }
        catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"parallel panels must require disk storage");
      }
      if(localized_cc::parallel::world().rank==0)
        std::cout << "Concurrent disk panel total/nM/restart gates passed\n";
      disk.maximum_scratch_bytes=1;
      bool rejected=false;
      try { (void)localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},{3,4,5},disk); }
      catch(const std::runtime_error&) { rejected=true; }
      require(rejected,"disk quota guard must fail collectively");
      require(std::filesystem::is_empty(directory.path),"failed disk plan created scratch files");
      disk.maximum_scratch_bytes=0;
      rejected=false;
      try { (void)localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},{3,4,5},disk); }
      catch(const std::invalid_argument&) { rejected=true; }
      require(rejected,"disk mode requires explicit nonzero budget");
      disk.maximum_scratch_bytes=1024*1024;
      disk.sector_streaming=false;
      rejected=false;
      try { (void)localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},{3,4,5},disk); }
      catch(const std::invalid_argument&) { rejected=true; }
      require(rejected,"disk mode requires sector streaming");
      disk.sector_streaming=true;
      disk.block_screening_budget_hartree=1.e-10;
      rejected=false;
      try { (void)localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},{3,4,5},disk); }
      catch(const std::invalid_argument&) { rejected=true; }
      require(rejected,"disk screening combination must fail closed");
      if(localized_cc::parallel::world().rank==0)
        std::cout << "Disk-backed triples total/M1-M6/panel/restart gates passed\n";
    }
#endif
    // Arbitrary partial point restart from the original executor, including
    // stale values in unfinished slots, must not double count sectors.
    compact.checkpoint_observer = {};
    compact.initial_point_many_body_energies = ref.point_many_body_energies;
    compact.completed_points = ref.completed_points;
    compact.completed_points[1] = 0;
    const auto resumed = localized_cc::solve_localized_triples_laplace_streamed(
        h,t1,t2,{0,1,2},{3,4,5},compact);
    require(resumed.restarted_points == opt.quadrature_points - 1, "compact partial restart count");
    for (std::size_t n=0;n<6;++n)
      close(resumed.many_body_energies[n],ref.many_body_energies[n],1.e-13,"compact restart M1-M6");
    compact.initial_point_many_body_energies.clear();
    compact.completed_points.clear();
    compact.maximum_additional_memory_bytes = low.planned_peak_additional_bytes - 1;
    bool rejected = false;
    try {
      (void)localized_cc::solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},{3,4,5},compact);
    } catch (const std::runtime_error&) { rejected = true; }
    require(rejected,"compact peak guard must fail closed");
  }
  for (const double budget:{1.e-10,1.e-7,1.e-5,1.0}) {
    opt.block_screening_budget_hartree=budget;
    const auto screened=localized_cc::solve_localized_triples_laplace_streamed(
        h,t1,t2,{0,1,2},{3,4,5},opt);
    require(screened.many_body_terms==ref.many_body_terms,"screen lost nM classes");
    require(screened.block_screening_bound_hartree<=budget,"six-label cumulative budget");
    for (std::size_t m=0;m<6;++m)
      close(screened.many_body_energies[m],ref.many_body_energies[m],
            screened.block_screening_bound_hartree+1.e-13,"M1-M6 screening bound");
    close(screened.parenthetical_energy,ref.parenthetical_energy,
          screened.block_screening_bound_hartree+1.e-13,"six-label total screening bound");
    if (budget==1.0)
      require(screened.block_screening_skipped>0,"six-label test did not exercise omission");
  }
}

} // namespace

void test_virtual_axis_slices() {
#if defined(__linux__)
  {
    TemporaryTriplesDirectory directory;
    const auto path=(directory.path/"atomic.chk").string();
    const std::vector<double> values{1.,-2.,3.};
    localized_cc::detail::write_sector_checkpoint(path,"test",values);
    require(localized_cc::detail::read_sector_checkpoint(path,"test",3)==values,"sector file roundtrip");
    { std::ofstream damaged(path,std::ios::binary|std::ios::app); damaged.put('x'); }
    bool rejected=false;
    try { (void)localized_cc::detail::read_sector_checkpoint(path,"test",3); }
    catch(const std::runtime_error&) { rejected=true; }
    require(rejected,"damaged sector file accepted");
    std::filesystem::remove(path);
  }
#endif
  const std::size_t n=3, n2=n*n;
  std::vector<double> source(n*n*n), p(n*n), first(source.size()), second(source.size());
  for(std::size_t i=0;i<source.size();++i) source[i]=(static_cast<double>(i)-12.)/17.;
  for(std::size_t i=0;i<p.size();++i) p[i]=(static_cast<double>(i)+1.)/19.;
  localized_cc::detail::transform_virtual_axes(source,p,n,first,second);
  for(std::size_t z=0;z<n;++z) for(std::size_t y=0;y<n;++y) for(std::size_t x=0;x<n;++x) {
    double expected=0.;
    for(std::size_t c=0;c<n;++c) for(std::size_t b=0;b<n;++b) for(std::size_t a=0;a<n;++a)
      expected+=p[x+n*a]*p[y+n*b]*p[z+n*c]*source[a+n*b+n2*c];
    close(first[x+n*y+n2*z],expected,1e-13,"virtual slice transform scalar oracle");
  }
  // Explicit NERSC gate only: exercise the failing W6 extent without moments,
  // disk files, integrals, or a CCSD solve. Rank-one input has an analytic result.
  if(std::getenv("LCCSDT_W6_VIRTUAL_AXIS_GATE")) {
    constexpr std::size_t m=216, square=m*m;
    std::vector<double> x(m), px(m), matrix(square), input(square*m);
    for(std::size_t i=0;i<m;++i) x[i]=static_cast<double>(i%11+1)/13.;
    for(std::size_t j=0;j<m;++j) for(std::size_t i=0;i<m;++i) {
      matrix[i+m*j]=(i==j ? .8 : .0001*static_cast<double>((i+j)%3));
      px[i]+=matrix[i+m*j]*x[j];
    }
    for(std::size_t k=0;k<m;++k) for(std::size_t j=0;j<m;++j) for(std::size_t i=0;i<m;++i)
      input[i+m*j+square*k]=x[i]*x[j]*x[k];
    int failures=0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(2) reduction(+:failures)
#endif
    for(int worker=0;worker<2;++worker) {
      std::vector<double> a(input.size()), b(input.size());
      localized_cc::detail::transform_virtual_axes(input,matrix,m,a,b);
      for(std::size_t k=0;k<m;++k) for(std::size_t j=0;j<m;++j) for(std::size_t i=0;i<m;++i) {
        const auto value=a[i+m*j+square*k];
        if(!std::isfinite(value)||std::abs(value-px[i]*px[j]*px[k])>1e-12) ++failures;
      }
    }
    require(failures==0,"W6 concurrent virtual slice analytic oracle");
    if(localized_cc::parallel::world().rank==0)
      std::cout << "W6 virtual-axis extent=216 analytic gate passed\n" << std::flush;
  }
}

void test_disk_occupied_panels() {
  for (const std::size_t extent : {std::size_t{4103}, std::size_t{1656360}}) {
    constexpr std::size_t sources=8;
    std::vector<double> input(sources*extent), coefficients(sources), packed;
    for(std::size_t k=0;k<sources;++k) {
      coefficients[k]=static_cast<double>(k+1)/19.;
      for(std::size_t v=0;v<extent;++v)
        input[k*extent+v]=(static_cast<double>((v+13*k)%97)-48.)/101.;
    }
    for(const std::size_t first : {std::size_t{0},extent-7}) {
      std::vector<double> output(std::min(std::size_t{4096},extent-first));
      localized_cc::detail::contract_disk_occupied_panel(input,extent,coefficients,first,output,packed);
      for(std::size_t v=0;v<output.size();++v) {
        double reference=0.;
        for(std::size_t k=0;k<sources;++k) reference+=coefficients[k]*input[k*extent+first+v];
        close(output[v],reference,1e-13,"disk occupied packed panel differs from scalar oracle");
      }
    }
    bool rejected=false;
    std::vector<double> output(8);
    try { localized_cc::detail::contract_disk_occupied_panel(input,extent,coefficients,extent-7,output,packed); }
    catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"disk occupied panel accepted an out-of-range tile");
  }
}

int main(int argc, char** argv) {
  localized_cc::parallel::Runtime parallel(argc, argv);
  try {
#if defined(__linux__)
    test_triples_row_file();
#endif
    test_legacy_one_body_rebuild();
    struct Triple { std::array<std::size_t, 3> spin; };
    const std::vector<Triple> triples{{{0, 1, 2}}, {{0, 2, 3}}, {{2, 4, 6}}};
    const localized_cc::detail::TripleMonomerPatterns patterns(triples, {7, 99, 105, 7});
    require(patterns.patterns.size() == 2 && patterns.columns[0] == patterns.columns[1], "compressed label sets");
    require(localized_cc::detail::triple_pattern_span({7, 99, -1}, {99, 105, -1}) == 3, "exact label union");
    require(localized_cc::detail::triple_pattern_span({0, 1, 2}, {3, 4, 5}) == 6, "M6 label union");
    // Exercise every bin, including M4/M5/M6, with repeated patterns and
    // signed contributions. Split rows must match one scalar reference pass.
    const std::array<double,8> left{1,-2,3,-4,5,-6,7,-8};
    const std::array<double,8> response{2,3,4,5,6,7,8,9};
    const std::array<std::size_t,8> ids{0,1,2,3,4,5,4,5};
    const std::array<unsigned char,6> orders{1,2,3,4,5,6};
    std::array<double,7> reference{}, actual{};
    for (std::size_t c=0;c<left.size();++c) {
      const double x=-0.5*left[c]*response[c];
      reference[ids[c]]+=x;
      reference[6]+=x;
    }
    for (std::size_t start=0;start<left.size();start+=3) {
      const auto length=std::min(std::size_t{3},left.size()-start);
      localized_cc::detail::accumulate_triples_energy_row(
          {left.data()+start,length},{response.data()+start,length},
          {ids.data()+start,length},orders.data(),0.5,actual);
    }
    for (std::size_t m=0;m<7;++m)
      close(actual[m],reference[m],0.0,"blocked M1--M6 scalar accumulation");
    test_laplace_resolvent_and_rotation_invariance();
    test_virtual_axis_slices();
    test_disk_occupied_panels();
    test_shared_moment_packing();
    test_screening_six_monomer_labels();
    if (localized_cc::parallel::world().rank == 0)
      std::cout << "Laplace localized triples tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    if (localized_cc::parallel::world().rank == 0)
      std::cerr << "Laplace triples test failure: " << error.what() << '\n';
    return 1;
  }
}
