#include "modernqc/mp2/laplace_fit.hpp"

#include "modernqc/linalg/eigensolver.hpp"
#include "modernqc/linalg/matrix.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef MODERNQC_HAS_MPI
#include <mpi.h>
#endif

#ifdef MODERNQC_HAS_OPENMP
#include <omp.h>
#endif

namespace modernqc::mp2 {
namespace {

constexpr double log_minimum_normal =
    -708.39641853226410622;

void validate_options(const LaplaceFitOptions& options) {
  if(options.histogram_bins < 2 || options.number_of_points == 0 ||
     options.number_of_points > options.histogram_bins ||
     options.gap_block_size == 0 || options.validation_grid_size < 2 ||
     options.maximum_optimizer_iterations == 0 || options.threads == 0 ||
     options.mpi_ranks == 0 || options.mpi_rank >= options.mpi_ranks ||
     options.threads >
         static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
     !std::isfinite(options.relative_singular_value_cutoff) ||
     options.relative_singular_value_cutoff <= 0.0 ||
     options.relative_singular_value_cutoff >= 1.0 ||
     !std::isfinite(options.normal_equation_condition_threshold) ||
     options.normal_equation_condition_threshold <= 1.0 ||
     !std::isfinite(options.initial_simplex_step) ||
     options.initial_simplex_step <= 0.0 ||
     !std::isfinite(options.logarithmic_node_tolerance) ||
     options.logarithmic_node_tolerance <= 0.0 ||
     !std::isfinite(options.objective_tolerance) ||
     options.objective_tolerance <= 0.0) {
    throw std::invalid_argument("Laplace-fit options are invalid");
  }
#ifndef MODERNQC_HAS_OPENMP
  if(options.threads != 1) {
    throw std::invalid_argument(
        "multiple histogram threads require OpenMP support");
  }
#endif
}

void validate_orbital_space(
    const std::vector<double>& orbital_energies,
    const std::vector<std::size_t>& active_occupied_indices,
    const std::vector<std::size_t>& virtual_indices) {
  if(orbital_energies.empty() || active_occupied_indices.empty() ||
     virtual_indices.empty()) {
    throw std::invalid_argument(
        "Laplace fitting requires nonempty energies and orbital spaces");
  }
  for(const double energy : orbital_energies) {
    if(!std::isfinite(energy)) {
      throw std::invalid_argument(
          "Laplace fitting received a nonfinite orbital energy");
    }
  }
  std::set<std::size_t> unique_indices;
  for(const auto* space : {&active_occupied_indices, &virtual_indices}) {
    for(const std::size_t orbital : *space) {
      if(orbital >= orbital_energies.size()) {
        throw std::invalid_argument(
            "Laplace-fitting orbital index is out of range");
      }
      if(!unique_indices.insert(orbital).second) {
        throw std::invalid_argument(
            "Laplace occupied/virtual indices overlap or repeat");
      }
    }
  }
}

[[nodiscard]] std::uint64_t checked_square_count(std::size_t count) {
  if(count >
     static_cast<std::size_t>(
         std::numeric_limits<std::uint64_t>::max() /
         static_cast<std::uint64_t>(count))) {
    throw std::overflow_error(
        "ordered Laplace denominator count exceeds uint64");
  }
  return static_cast<std::uint64_t>(count) *
         static_cast<std::uint64_t>(count);
}

[[nodiscard]] DenominatorHistogram build_denominator_interval(
    const std::vector<double>& orbital_energies,
    const std::vector<std::size_t>& active_occupied_indices,
    const std::vector<std::size_t>& virtual_indices,
    const LaplaceFitOptions& options) {
  validate_options(options);
  validate_orbital_space(orbital_energies, active_occupied_indices,
                         virtual_indices);

  double occupied_minimum = std::numeric_limits<double>::infinity();
  double occupied_maximum = -std::numeric_limits<double>::infinity();
  for(const std::size_t orbital : active_occupied_indices) {
    occupied_minimum =
        std::min(occupied_minimum, orbital_energies[orbital]);
    occupied_maximum =
        std::max(occupied_maximum, orbital_energies[orbital]);
  }
  double virtual_minimum = std::numeric_limits<double>::infinity();
  double virtual_maximum = -std::numeric_limits<double>::infinity();
  for(const std::size_t orbital : virtual_indices) {
    virtual_minimum =
        std::min(virtual_minimum, orbital_energies[orbital]);
    virtual_maximum =
        std::max(virtual_maximum, orbital_energies[orbital]);
  }
  const double denominator_minimum =
      2.0 * virtual_minimum - 2.0 * occupied_maximum;
  const double denominator_maximum =
      2.0 * virtual_maximum - 2.0 * occupied_minimum;
  if(!std::isfinite(denominator_minimum) ||
     !std::isfinite(denominator_maximum) ||
     denominator_minimum <= 0.0 ||
     denominator_maximum <= denominator_minimum) {
    throw std::runtime_error(
        "Laplace denominator interval is nonpositive or degenerate");
  }
  if(active_occupied_indices.size() >
     std::numeric_limits<std::size_t>::max() / virtual_indices.size()) {
    throw std::overflow_error("occupied-virtual gap count overflows size_t");
  }
  const std::size_t gap_count =
      active_occupied_indices.size() * virtual_indices.size();
  return DenominatorHistogram{
      .denominator_minimum = denominator_minimum,
      .denominator_maximum = denominator_maximum,
      .total_ordered_denominators = checked_square_count(gap_count),
      .edges = {denominator_minimum, denominator_maximum},
      .midpoints = {},
      .counts = {},
      .frequencies = {},
  };
}

[[nodiscard]] double safe_decay(double denominator, double node) {
  const double exponent = -denominator * node;
  if(!std::isfinite(exponent)) {
    throw std::runtime_error(
        "Laplace exponential argument is nonfinite");
  }
  if(exponent < log_minimum_normal) {
    return 0.0;
  }
  return std::exp(exponent);
}

void reduce_histogram_across_ranks(
    std::vector<std::uint64_t>& counts,
    const LaplaceFitOptions& options) {
  if(options.mpi_ranks == 1) {
    return;
  }
#ifdef MODERNQC_HAS_MPI
  int initialized = 0;
  int rank = 0;
  int ranks = 0;
  if(MPI_Initialized(&initialized) != MPI_SUCCESS || initialized == 0 ||
     MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
     MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS ||
     rank < 0 || ranks < 1 ||
     static_cast<std::size_t>(rank) != options.mpi_rank ||
     static_cast<std::size_t>(ranks) != options.mpi_ranks) {
    throw std::runtime_error(
        "Laplace histogram MPI topology does not match MPI_COMM_WORLD");
  }
  std::size_t offset = 0;
  while(offset < counts.size()) {
    const std::size_t remaining = counts.size() - offset;
    const int chunk = static_cast<int>(
        std::min(remaining,
                 static_cast<std::size_t>(
                     std::numeric_limits<int>::max())));
    if(MPI_Allreduce(MPI_IN_PLACE, counts.data() + offset, chunk,
                     MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
      throw std::runtime_error(
          "MPI_Allreduce failed for Laplace denominator histogram");
    }
    offset += static_cast<std::size_t>(chunk);
  }
#else
  (void)counts;
  throw std::runtime_error(
      "multiple Laplace histogram ranks require MPI support");
#endif
}

#ifdef MODERNQC_HAS_MPI
void broadcast_bytes(void* data, std::size_t bytes) {
  auto* cursor = static_cast<unsigned char*>(data);
  std::size_t offset = 0;
  while(offset < bytes) {
    const int chunk = static_cast<int>(
        std::min(bytes - offset,
                 static_cast<std::size_t>(
                     std::numeric_limits<int>::max())));
    if(MPI_Bcast(cursor + offset, chunk, MPI_BYTE, 0,
                 MPI_COMM_WORLD) != MPI_SUCCESS) {
      throw std::runtime_error("MPI_Bcast failed for Laplace fit data");
    }
    offset += static_cast<std::size_t>(chunk);
  }
}

void broadcast_vector(std::vector<double>& values) {
  if(!values.empty()) {
    if(values.size() >
       std::numeric_limits<std::size_t>::max() / sizeof(double)) {
      throw std::overflow_error(
          "Laplace fit vector byte count overflows size_t");
    }
    broadcast_bytes(values.data(), values.size() * sizeof(double));
  }
}

[[nodiscard]] std::size_t checked_size(std::uint64_t value,
                                       const char* description) {
  if(value > static_cast<std::uint64_t>(
                 std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error(std::string{description} +
                              " exceeds size_t");
  }
  return static_cast<std::size_t>(value);
}

[[nodiscard]] LaplaceFitResult root_fit_and_broadcast(
    const DenominatorHistogram& histogram,
    const LaplaceFitOptions& options) {
  std::optional<LaplaceFitResult> root_result;
  std::string root_error;
  if(options.mpi_rank == 0) {
    try {
      root_result = fit_laplace_quadrature(histogram, options);
    } catch(const std::exception& error) {
      root_error = error.what();
    } catch(...) {
      root_error = "unknown rank-zero Laplace fit failure";
    }
  }
  int failed = root_error.empty() ? 0 : 1;
  if(MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Bcast failed for rank-zero Laplace fit status");
  }
  if(failed != 0) {
    std::uint64_t length =
        options.mpi_rank == 0
            ? static_cast<std::uint64_t>(root_error.size())
            : 0;
    if(MPI_Bcast(&length, 1, MPI_UINT64_T, 0,
                 MPI_COMM_WORLD) != MPI_SUCCESS) {
      throw std::runtime_error(
          "MPI_Bcast failed for Laplace fit error length");
    }
    if(options.mpi_rank != 0) {
      root_error.resize(checked_size(length, "Laplace fit error length"));
    }
    if(length != 0) {
      broadcast_bytes(root_error.data(), root_error.size());
    }
    throw std::runtime_error("rank-zero Laplace fit failed: " + root_error);
  }

  std::array<std::uint64_t, 5> sizes{};
  std::array<double, 6> diagnostics{};
  if(options.mpi_rank == 0) {
    sizes = {
        static_cast<std::uint64_t>(root_result->nodes.size()),
        static_cast<std::uint64_t>(
            root_result->optimization_history.size()),
        static_cast<std::uint64_t>(
            root_result->condition_number_history.size()),
        static_cast<std::uint64_t>(root_result->optimizer_iterations),
        static_cast<std::uint64_t>(root_result->weight_numerical_rank),
    };
    diagnostics = {
        root_result->weighted_objective,
        root_result->weighted_rmse,
        root_result->histogram_maximum_absolute_error,
        root_result->histogram_maximum_relative_error,
        root_result->validation_maximum_absolute_error,
        root_result->validation_maximum_relative_error,
    };
  }
  if(MPI_Bcast(sizes.data(), static_cast<int>(sizes.size()), MPI_UINT64_T,
               0, MPI_COMM_WORLD) != MPI_SUCCESS ||
     MPI_Bcast(diagnostics.data(), static_cast<int>(diagnostics.size()),
               MPI_DOUBLE, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Bcast failed for Laplace fit metadata");
  }
  if(options.mpi_rank != 0) {
    const std::size_t node_count =
        checked_size(sizes[0], "Laplace node count");
    root_result = LaplaceFitResult{
        .histogram = histogram,
        .nodes = std::vector<double>(node_count),
        .weights = std::vector<double>(node_count),
        .optimization_history = std::vector<double>(
            checked_size(sizes[1], "Laplace objective-history length")),
        .condition_number_history = std::vector<double>(
            checked_size(sizes[2], "Laplace condition-history length")),
        .optimizer_iterations =
            checked_size(sizes[3], "Laplace optimizer iteration count"),
        .weight_numerical_rank =
            checked_size(sizes[4], "Laplace numerical rank"),
        .weighted_objective = diagnostics[0],
        .weighted_rmse = diagnostics[1],
        .histogram_maximum_absolute_error = diagnostics[2],
        .histogram_maximum_relative_error = diagnostics[3],
        .validation_maximum_absolute_error = diagnostics[4],
        .validation_maximum_relative_error = diagnostics[5],
    };
  }
  broadcast_vector(root_result->nodes);
  broadcast_vector(root_result->weights);
  broadcast_vector(root_result->optimization_history);
  broadcast_vector(root_result->condition_number_history);
  return std::move(*root_result);
}
#endif

struct WeightFit {
  std::vector<double> nodes;
  std::vector<double> weights;
  double objective;
  double condition_number;
  std::size_t numerical_rank;
};

void error_diagnostics(const std::vector<double>& denominators,
                       const std::vector<double>& nodes,
                       const std::vector<double>& weights,
                       double& maximum_absolute,
                       double& maximum_relative);

struct MinimaxParameters {
  std::vector<double> nodes;
  std::vector<double> weights;
  double alternating_error{0.0};
  double jacobian_condition_number{1.0};
};

template<std::size_t Points>
struct MinimaxSeed {
  double interval_ratio;
  std::array<double, Points> weights;
  std::array<double, Points> nodes;
};

// Pretabulated coefficients provide the Newton-Remez starting basin described
// by Takatsuka, Ten-no, and Hackbusch (JCP 129, 044112, 2008). They are start
// values only: the requested denominator interval is optimized below.
constexpr std::array<MinimaxSeed<5>, 18> five_point_minimax_seeds{{
    {2.0, {0.4683556115061091, 1.1387420740801897,
           1.9531818897084401, 3.1153038431409947,
           5.3979418510235631},
          {0.18161223444789812, 0.97768304151957064,
           2.5060472629612187, 4.9959238094351379,
           9.0782980472197039}},
    {3.0, {0.36871557779169384, 0.90701795181963019,
           1.5911924263942965, 2.6280030891313686,
           4.7716680846893071},
          {0.14278665283325309, 0.77316533723765379,
           2.0043671769709315, 4.0691881695563703,
           7.5990884376890246}},
    {4.0, {0.30850461747146762, 0.76778306449950773,
           1.3777178717301291, 2.3531329292269723,
           4.4453037966406379},
          {0.11931452368959949, 0.64982392840005032,
           1.7039690954687403, 3.5232332927125776,
           6.75309425511802}},
    {5.0, {0.26751449382571008, 0.6731648185410648,
           1.2341495375414908, 2.1730749987606481,
           4.240049104595772},
          {0.10333512117595546, 0.56589985547987387,
           1.5002594864446179, 3.156388982390542,
           6.1935841722085909}},
    {6.0, {0.23752137119510233, 0.60395287839727363,
           1.1298096471277477, 2.0442936920871437,
           4.0963491911779881},
          {0.091644428491289509, 0.5044896950537755,
           1.3514626526197429, 2.8899298640154387,
           5.7907824263917469}},
    {7.0, {0.2144769745022462, 0.55075851621144056,
           1.0499538293945301, 1.9466601062734827,
           3.98859281695391},
          {0.082664038655757152, 0.45729486145269549,
           1.2372195309898968, 2.6860512679499591,
           5.484100671224871}},
    {8.0, {0.19613395743207157, 0.50839048356373462,
           0.98652280495664868, 1.8695023711995027,
           3.9038727784001863},
          {0.075517387268414365, 0.41971519861212586,
           1.1462958376485461, 2.5241149420764288,
           5.2411231041616579}},
    {9.0, {0.18113555675752338, 0.4737218817910962,
           0.93470388344661171, 1.8066027816920285,
           3.8349340375191976},
          {0.069675180489385216, 0.3889757291641015,
           1.0719353806517393, 2.3918121851951577,
           5.0428035352623768}},
    {10.0, {0.16861003324210627, 0.44474622263913965,
            0.8914300222454683, 1.7540776690871602,
            3.777359367020908},
           {0.064797316294974178, 0.36329423979113412,
            1.009808499526931, 2.2813077174805523,
            4.8771539519428151}},
    {20.0, {0.10434198456045442, 0.29555856650050433,
            0.66783133886464374, 1.4788816616452081,
            3.4728708916802091},
           {0.039791035518631973, 0.23131332253827694,
            0.68992455340431735, 1.7098435082055961,
            4.0149225768154402}},
    {30.0, {0.078520229672523728, 0.2352291127246095,
            0.57552354590383326, 1.3603613273128701,
            3.3389033426820007},
           {0.029757786779991018, 0.17814645562861608,
            0.56011388529387329, 1.4739028120126259,
            3.6519366545145373}},
    {40.0, {0.064181483805067871, 0.20155531773488039,
            0.52269108794031494, 1.2901011954205592,
            3.2584115181337521},
           {0.024190180350097283, 0.14857337492633746,
            0.48727597503891512, 1.3391490990318979,
            3.4411575727997534}},
    {50.0, {0.054928968471960854, 0.17972260846046623,
            0.48760926843571956, 1.2421875649935035,
            3.2030368198904093},
           {0.020598516936418058, 0.12946507144298772,
            0.43979723967050899, 1.2499372840249323,
            3.2997722744581939}},
    {60.0, {0.048409239447676798, 0.16426671642208504,
            0.4622317543222329, 1.2068045801673655,
            3.1618887677979015},
           {0.018067709913646333, 0.11598435008616298,
            0.40601702895000163, 1.1856093034676984,
            3.1967392715623495}},
    {70.0, {0.04354046414237496, 0.15267140778657548,
            0.44282189639585939, 1.1792923812761709,
            3.129743372172042},
           {0.016177447463648808, 0.10590557466186795,
            0.38055858744901055, 1.1365624686784197,
            3.1174919552876026}},
    {80.0, {0.039751433153653738, 0.14360627651378796,
            0.42738350642551687, 1.1571122334202288,
            3.1037318963686116},
           {0.014705949753103624, 0.098052868167850485,
            0.36057332806560449, 1.0976668403036145,
            3.0541811446566616}},
    {90.0, {0.036710311593398881, 0.13629750852526029,
            0.41474261404916296, 1.1387451937085928,
            3.0821271502266367},
           {0.013524447064889847, 0.091742821460122956,
            0.34440049009587181, 1.0659073186356656,
            3.0021576019667893}},
    {100.0, {0.034210303487463151, 0.1302622960820673,
             0.40415841564587901, 1.1232181675451096,
             3.0638169728620763},
            {0.012552727540467989, 0.086549394877289257,
             0.33100124293713873, 1.0393832416814095,
             2.9584699518773347}},
}};

constexpr std::array<MinimaxSeed<6>, 18> six_point_minimax_seeds{{
    {2.0,
     {0.3951503586666183, 0.94835848039535586, 1.5813287669193052,
      2.3797403876283134, 3.5481339355732611, 5.8817260741536765},
     {0.15344232732696164, 0.82080301441425429, 2.0763266377595739,
      4.0375449448252487, 6.9549672561485965, 11.489101449975879}},
    {3.0,
     {0.31060826041851003, 0.75140935613433568, 1.2720076125413839,
      1.9595162535981736, 3.0177199430463202, 5.2126406909045242},
     {0.12050409838853515, 0.64717063747638337, 1.6495023836455649,
      3.2457669505020714, 5.6881108969184009, 9.6288261179194556}},
    {4.0,
     {0.25949253370837799, 0.63269942815416391, 1.0873449642129085,
      1.7143811392552479, 2.7218526270988348, 4.8651609154509243},
     {0.10058381094183846, 0.5423052006448188, 1.39268940689611,
      2.7733849875599961, 4.9443770209132838, 8.5653072999212299}},
    {5.0,
     {0.22469037674868492, 0.55190958441428917, 0.96220193942661358,
      1.5504627855961601, 2.5290208569930264, 4.6466141585747067},
     {0.087021634064755576, 0.47091367520479227, 1.2180676458000685,
      2.4535517473926842, 4.4452949405703528, 7.8614295814483421}},
    {6.0,
     {0.19922622623379935, 0.49276258797330597, 0.87076495074419136,
      1.431682309041832, 2.391339554164293, 4.4933628695491539},
     {0.077099897501371778, 0.41866258611030904, 1.0902916798550335,
      2.2200677428084221, 4.0828499947174546, 7.3540700896552895}},
    {7.0,
     {0.17966376455572852, 0.44727832190667599, 0.80050822414634848,
      1.3408813762157024, 2.2869350476400565, 4.3781845743582668},
     {0.069479122974795818, 0.3785036647871915, 0.99206290357187255,
      2.0408032638639466, 3.8053977452053824, 6.967212719012414}},
    {8.0,
     {0.16409461040061746, 0.41103624081412404, 0.74453715494776329,
      1.268755832367187, 2.2043138776461757, 4.2874025674581784},
     {0.063415234190309311, 0.34652627576455186, 0.91380939058785049,
      1.8980781549742471, 3.5848315846970733, 6.6602309134426045}},
    {9.0,
     {0.1513663895024808, 0.38137058330649815, 0.69871151354338523,
      1.2097848806132692, 2.1368239386439773, 4.2133423984548939},
     {0.058458869963323397, 0.32037011449359309, 0.84976328784135302,
      1.7812813101063347, 3.4044265671848599, 6.4092678952397231}},
    {10.0,
     {0.1407385838051434, 0.35656942006856901, 0.66038026167380126,
      1.1604682915072342, 2.0803301182531366, 4.1513349861478135},
     {0.054321259850005937, 0.29851876473581168, 0.7962225930216239,
      1.6836229524774231, 3.2535534765211933, 6.199310760184936}},
    {20.0,
     {0.086249674118003536, 0.22875523138387599, 0.46192316534351391,
      0.90291379158982632, 1.7807264424526321, 3.8200570280772443},
     {0.033125384072048854, 0.18624531899278388, 0.52022621247932066,
      1.1783399761432671, 2.468203283261333, 5.0989823083682602}},
    {30.0,
     {0.064387539226077442, 0.17699255332037328, 0.38039893140000219,
      0.7936758834219183, 1.6486087888192316, 3.671677753256644},
     {0.024633860186806189, 0.14102557634568977, 0.4082545329312478,
      0.97074992752334099, 2.1393343031375327, 4.6296955440744574}},
    {40.0,
     {0.052257474234297809, 0.14807969939302706, 0.33413952929595248,
      0.72969218759311816, 1.5689274489411984, 3.5813517206284184},
     {0.019927292158899282, 0.11586876096539546, 0.345547325025503,
      0.85291869643996032, 1.949300779528476, 4.3543982689127896}},
    {50.0,
     {0.044433418238675776, 0.12933083480399385, 0.30367117547648526,
      0.68639853280859997, 1.513850308723576, 3.5185424210031164},
     {0.016893837552218283, 0.099609177768238424, 0.3047652371799654,
      0.77531837694853878, 1.8222505520282237, 4.1681458423137245}},
    {60.0,
     {0.038920753921461437, 0.11606060550997924, 0.28178279580412929,
      0.65458647059582142, 1.4727168297337245, 3.4714354715327134},
     {0.014757794449453762, 0.088133968050174691, 0.27581359687169688,
      0.71959688160612523, 1.7298527418488276, 4.0313890355775026}},
    {70.0,
     {0.034803217566193485, 0.10610858572603164, 0.26513597229761221,
      0.62992689456642181, 1.4404181621079915, 3.4343276793451509},
     {0.013163067275079546, 0.079550705631287605, 0.25403700481526015,
      0.67724944380318797, 1.6588625672007167, 3.9254796374566991}},
    {80.0,
     {0.031597334451890974, 0.098331250459754174, 0.25195493890390575,
      0.61008071138317543, 1.4141485422756008, 3.404069746353219},
     {0.011921868297062481, 0.07285946900570793, 0.23697015769013335,
      0.64374794964932713, 1.6021676951562196, 3.8403261825125701}},
    {90.0,
     {0.029022435615502142, 0.092062996251443749, 0.24120038626341075,
      0.59365817930113973, 1.3922177274028811, 3.3787565044853691},
     {0.010925238490803518, 0.0674791066323321, 0.22317706781392338,
      0.61644017362727033, 1.5555685911402399, 3.7699289128884463}},
    {100.0,
     {0.026903750347277787, 0.086888403464698963, 0.23221984689982497,
      0.57977446662637944, 1.3735366550622796, 3.3571561544965229},
     {0.010105361344629787, 0.063047379303821371, 0.21176070087633833,
      0.59366012072477725, 1.5164074607971301, 3.7104671222364463}},
}};

constexpr std::array<MinimaxSeed<7>, 18> seven_point_minimax_seeds{{
    {2.0,
     {0.34183027419014206, 0.81393739758765016, 1.3352973461439932,
      1.9505008993945934, 2.744440392392776, 3.9247848246178934,
      6.3099568403220472},
     {0.13285379559987881, 0.70791302191166172, 1.7769359317465696,
      3.4093982762602817, 5.7364147937170298, 9.0227759271360473,
      13.953318359742749}},
    {3.0,
     {0.2684261108739403, 0.64284943010975126, 1.0661026045730193,
      1.5833307838749986, 2.2798283994607749, 3.3594350485176023,
      5.6037743686535748},
     {0.10425591119922591, 0.55712995628619155, 1.4060269600791764,
      2.7202669793336032, 4.631373848989341, 7.4035412706077315,
      11.706208031960966}},
    {4.0,
     {0.22402959509378434, 0.53956884312067022, 0.90449191471747414,
      1.3657443364559092, 2.0115246527093902, 3.0469664839439665,
      5.2381022966734241},
     {0.086956305134267553, 0.46599558018018294, 1.1823447473116389,
      2.3066801208193026, 3.9742794392437286, 6.4556528129586557,
      10.422112499317336}},
    {5.0,
     {0.19380100796794852, 0.4692435586300529, 0.7946376596686956,
      1.2188241997775222, 1.8330663504453952, 2.8441417219183061,
      5.0081428000286365},
     {0.075178012053935581, 0.40394057377108084, 1.0300953839123785,
      2.0257208844822334, 3.5300416178849785, 5.8202600945046035,
      9.5718793022987914}},
    {6.0,
     {0.17168478698145412, 0.41775127828336472, 0.71421787638474932,
      1.1116696392634222, 1.7040902635819437, 2.6995153646546877,
      4.8467140323975917},
     {0.066561763985874725, 0.35852406376155277, 0.91863344456466633,
      1.8201812078424351, 3.2059208290570349, 5.3588978962327198,
      8.9585138516334126}},
    {7.0,
     {0.15469632923142762, 0.37815536501928487, 0.65234571194404356,
      1.0293972455516656, 1.6055887318778359, 2.5898198883040795,
      4.725197806343358},
     {0.059944311849698062, 0.32362223949285779, 0.83292407832976423,
      1.6621487568152902, 2.9570712920382243, 5.0055969194967371,
      8.4903507691443831}},
    {8.0,
     {0.14117768906298914, 0.34660927676840281, 0.60300882902484199,
      0.96385681284750846, 1.5273335779508246, 2.5029176530161652,
      4.6292517534532687},
     {0.054679354700405088, 0.29583554119399269, 0.76463593118796591,
      1.5362084580691611, 2.7588796360606658, 4.7245451363070368,
      8.1184429696940921}},
    {9.0,
     {0.13012755743255019, 0.32079191733729989, 0.56258820139654608,
      0.91017275757170202, 1.4632925602129687, 2.4318199116503538,
      4.5508383568435686},
     {0.050376531687526448, 0.27311145827271915, 0.70874389673528193,
      1.4330831595065083, 2.596601846366239, 4.4944705710233963,
      7.8140604336611439}},
    {10.0,
     {0.12090244093952099, 0.29921214711072491, 0.52876236081329564,
      0.8652325406126139, 1.4096599453909606, 2.3721982300004751,
      4.4850705608595405},
     {0.046784971919695058, 0.25413109177030713, 0.66202063661281685,
      1.3468236945914096, 2.4608175652259985, 4.3018718689144126,
      7.559127174440305}},
    {20.0,
     {0.073640943793741087, 0.18809607758877431, 0.35358561229255181,
      0.63102636982655691, 1.1268723268357688, 2.0532076958089367,
      4.1312146140908652},
     {0.02839787501935781, 0.15669465451979245, 0.42128353664098478,
      0.90071033948500667, 1.7551920884002583, 3.2943879648814645,
      6.2167521169331161}},
    {30.0,
     {0.05470812366039994, 0.14316733618034405, 0.2818519061350474,
      0.5330147207850684, 1.0042678371684879, 1.9102105077080431,
      3.9707839278055399},
     {0.021041597496575321, 0.11751591067464534, 0.32378519947192319,
      0.71825110743343679, 1.4622083631638707, 2.8681290607978678,
      5.6391282837140384}},
    {40.0,
     {0.044216344397515227, 0.11809953230597657, 0.24136520239356096,
      0.47639520405533553, 0.93120652311733532, 1.8229592256705782,
      3.8722630501939475},
     {0.016968944347647359, 0.09574540214390341, 0.2692943329972381,
      0.6152724295093569, 1.2943422432026803, 2.6197801492130472,
      5.2979645334238246}},
    {50.0,
     {0.037455686176677001, 0.10185840534484672, 0.21485110258922235,
      0.43851262496240967, 0.88110126016728485, 1.7621025260651733,
      3.8032676122312763},
     {0.01434661695411904, 0.081686636965124057, 0.23392918256659598,
      0.54781736961611505, 1.1828967187208399, 2.4526048384096861,
      5.0658409539581051}},
    {60.0,
     {0.032696077420972573, 0.090372087405865345, 0.19590828145261074,
      0.41092391175017151, 0.84388006003205818, 1.7163114229596177,
      3.751205173302536},
     {0.012501624790553638, 0.071771232105286686, 0.20887384143197713,
      0.49961354804651559, 1.1023091213725187, 2.3303116968910063,
      4.8945625765439793}},
    {70.0,
     {0.029143346494564064, 0.081764233542750073, 0.18157540048038484,
      0.38969019728028792, 0.8147620352906384, 1.6801229596135674,
      3.7099722173278535},
     {0.011125203816029645, 0.064358601497210413, 0.1900644749502538,
      0.46313529471763043, 1.0406826317548055, 2.2358669464308134,
      4.7613343074885286}},
    {80.0,
     {0.02637871732005645, 0.075042150908638222, 0.17027978818667666,
      0.37269963545122914, 0.79114197546316622, 1.6505212211957283,
      3.6761865418821604},
     {0.010054615399715751, 0.058582476483652168, 0.17535029502975125,
      0.43438561313685364, 0.99165728131881903, 2.1600909301499334,
      4.6537869270461902}},
    {90.0,
     {0.024159221705021952, 0.069628219469392336, 0.16110291618092115,
      0.35870654013259184, 0.77146045304267763, 1.6256811668432707,
      3.6477955931137629},
     {0.0091954804054835866, 0.053939660471890652, 0.16347931911580282,
      0.4110293908202165, 0.9514935033869456, 2.0975458645195082,
      4.5645482583921666}},
    {100.0,
     {0.022333642551176373, 0.065162010730403855, 0.15346948447776387,
      0.34692296202841277, 0.75471793703896095, 1.6044221299260411,
      3.6234684847661183},
     {0.0084890821340208086, 0.050116642014961708, 0.15366998285929265,
      0.39160355318450513, 0.91783364668505174, 2.0447798621093103,
      4.4889127432163773}},
}};

template<std::size_t Points>
[[nodiscard]] MinimaxParameters select_minimax_seed(
    const std::array<MinimaxSeed<Points>, 18>& seeds, double ratio) {
  if(!std::isfinite(ratio) || ratio < 2.0 || ratio > 100.0) {
    throw std::invalid_argument(
        "minimax denominator ratio must be in [2, 100]");
  }
  const MinimaxSeed<Points>* selected = &seeds.front();
  for(const MinimaxSeed<Points>& candidate : seeds) {
    if(candidate.interval_ratio > ratio) {
      break;
    }
    selected = &candidate;
  }
  return MinimaxParameters{
      .nodes = {selected->nodes.begin(), selected->nodes.end()},
      .weights = {selected->weights.begin(), selected->weights.end()},
  };
}

template<std::size_t Points>
[[nodiscard]] MinimaxParameters select_enclosing_minimax_seed(
    const std::array<MinimaxSeed<Points>, 18>& seeds, double ratio) {
  if(!std::isfinite(ratio) || ratio < 2.0 || ratio > 100.0) {
    throw std::invalid_argument(
        "minimax denominator ratio must be in [2, 100]");
  }
  const MinimaxSeed<Points>* selected = &seeds.back();
  for(const MinimaxSeed<Points>& candidate : seeds) {
    if(candidate.interval_ratio >= ratio) {
      selected = &candidate;
      break;
    }
  }
  return MinimaxParameters{
      .nodes = {selected->nodes.begin(), selected->nodes.end()},
      .weights = {selected->weights.begin(), selected->weights.end()},
  };
}

[[nodiscard]] MinimaxParameters minimax_seed(std::size_t points,
                                             double ratio) {
  switch(points) {
    case 5:
      return select_minimax_seed(five_point_minimax_seeds, ratio);
    case 6:
      return select_minimax_seed(six_point_minimax_seeds, ratio);
    case 7:
      return select_minimax_seed(seven_point_minimax_seeds, ratio);
    default:
      throw std::invalid_argument(
          "the current minimax implementation supports 5, 6, or 7 "
          "quadrature points");
  }
}

[[nodiscard]] MinimaxParameters enclosing_minimax_seed(
    std::size_t points, double ratio) {
  switch(points) {
    case 6:
      return select_enclosing_minimax_seed(
          six_point_minimax_seeds, ratio);
    case 7:
      return select_enclosing_minimax_seed(
          seven_point_minimax_seeds, ratio);
    default:
      throw std::invalid_argument(
          "enclosing minimax seeds are available for 6 or 7 "
          "quadrature points");
  }
}

[[nodiscard]] double minimax_error(
    double denominator, const MinimaxParameters& parameters) {
  return evaluate_laplace_reciprocal(
             denominator, parameters.nodes, parameters.weights) -
         1.0 / denominator;
}

[[nodiscard]] double minimax_error_derivative(
    double denominator, const MinimaxParameters& parameters) {
  long double derivative =
      1.0L / (static_cast<long double>(denominator) *
              static_cast<long double>(denominator));
  for(std::size_t point = 0; point < parameters.nodes.size(); ++point) {
    derivative -=
        static_cast<long double>(parameters.weights[point]) *
        static_cast<long double>(parameters.nodes[point]) *
        static_cast<long double>(
            safe_decay(denominator, parameters.nodes[point]));
  }
  return static_cast<double>(derivative);
}

[[nodiscard]] std::vector<double> initial_minimax_alternant(
    double ratio, std::size_t points,
    const MinimaxParameters& parameters) {
  const std::size_t required_interior = 2 * points - 1;
  const std::size_t grid_size = std::max<std::size_t>(10000, 2000 * points);
  const double logarithmic_ratio = std::log(ratio);
  std::vector<double> roots;
  roots.reserve(required_interior);
  double lower = 1.0;
  double lower_derivative = minimax_error_derivative(lower, parameters);
  for(std::size_t grid_point = 1; grid_point <= grid_size; ++grid_point) {
    const double upper = std::exp(
        logarithmic_ratio * static_cast<double>(grid_point) /
        static_cast<double>(grid_size));
    const double upper_derivative =
        minimax_error_derivative(upper, parameters);
    if(lower_derivative == 0.0 ||
       std::signbit(lower_derivative) != std::signbit(upper_derivative)) {
      double bracket_lower = lower;
      double bracket_upper = upper;
      double bracket_value = lower_derivative;
      for(std::size_t iteration = 0; iteration < 100; ++iteration) {
        const double midpoint = std::exp(
            0.5 * (std::log(bracket_lower) +
                   std::log(bracket_upper)));
        const double midpoint_derivative =
            minimax_error_derivative(midpoint, parameters);
        if(midpoint_derivative == 0.0 ||
           std::abs(std::log(bracket_upper / bracket_lower)) <= 2.0e-14) {
          bracket_lower = midpoint;
          bracket_upper = midpoint;
          break;
        }
        if(std::signbit(bracket_value) ==
           std::signbit(midpoint_derivative)) {
          bracket_lower = midpoint;
          bracket_value = midpoint_derivative;
        } else {
          bracket_upper = midpoint;
        }
      }
      const double root = std::exp(
          0.5 * (std::log(bracket_lower) +
                 std::log(bracket_upper)));
      if(root > 1.0 && root < ratio &&
         (roots.empty() ||
          std::abs(std::log(root / roots.back())) > 1.0e-10)) {
        roots.push_back(root);
      }
    }
    lower = upper;
    lower_derivative = upper_derivative;
  }
  while(roots.size() < required_interior) {
    std::vector<double> boundaries;
    boundaries.reserve(roots.size() + 2);
    boundaries.push_back(1.0);
    boundaries.insert(boundaries.end(), roots.begin(), roots.end());
    boundaries.push_back(ratio);
    std::size_t largest_gap = 0;
    double largest_logarithmic_width = -1.0;
    for(std::size_t interval = 0; interval + 1 < boundaries.size();
        ++interval) {
      const double width =
          std::log(boundaries[interval + 1] / boundaries[interval]);
      if(width > largest_logarithmic_width) {
        largest_logarithmic_width = width;
        largest_gap = interval;
      }
    }
    roots.push_back(std::sqrt(boundaries[largest_gap] *
                              boundaries[largest_gap + 1]));
    std::sort(roots.begin(), roots.end());
  }
  if(roots.size() != required_interior) {
    throw std::runtime_error(
        "minimax initializer did not produce the required alternant; "
        "interior extrema=" + std::to_string(roots.size()) +
        " required=" + std::to_string(required_interior));
  }
  std::vector<double> alternant;
  alternant.reserve(2 * points + 1);
  alternant.push_back(1.0);
  alternant.insert(alternant.end(), roots.begin(), roots.end());
  alternant.push_back(ratio);
  return alternant;
}

void sort_minimax_node_weight_pairs(MinimaxParameters& parameters) {
  std::vector<std::pair<double, double>> pairs;
  pairs.reserve(parameters.nodes.size());
  for(std::size_t point = 0; point < parameters.nodes.size(); ++point) {
    pairs.emplace_back(parameters.nodes[point], parameters.weights[point]);
  }
  std::sort(pairs.begin(), pairs.end());
  for(std::size_t point = 0; point < pairs.size(); ++point) {
    parameters.nodes[point] = pairs[point].first;
    parameters.weights[point] = pairs[point].second;
  }
}

[[nodiscard]] double adjacent_alternation_residual(
    const std::vector<double>& alternant,
    const MinimaxParameters& parameters,
    std::vector<double>* residual = nullptr) {
  double maximum = 0.0;
  if(residual != nullptr) {
    residual->assign(alternant.size() - 1, 0.0);
  }
  for(std::size_t index = 0; index + 1 < alternant.size(); ++index) {
    const double value = minimax_error(alternant[index], parameters) +
                         minimax_error(alternant[index + 1], parameters);
    maximum = std::max(maximum, std::abs(value));
    if(residual != nullptr) {
      (*residual)[index] = value;
    }
  }
  return maximum;
}

[[nodiscard]] MinimaxParameters solve_minimax_alternation(
    const std::vector<double>& alternant,
    MinimaxParameters parameters, const LaplaceFitOptions& options) {
  const std::size_t points = parameters.nodes.size();
  const std::size_t dimension = 2 * points;
  if(alternant.size() != dimension + 1 ||
     parameters.weights.size() != points) {
    throw std::invalid_argument(
        "minimax alternation dimensions are inconsistent");
  }
  constexpr std::size_t maximum_newton_iterations = 300;
  double final_residual = std::numeric_limits<double>::infinity();
  double final_target = 0.0;
  for(std::size_t iteration = 0; iteration < maximum_newton_iterations;
      ++iteration) {
    std::vector<double> residual;
    const double current_residual = adjacent_alternation_residual(
        alternant, parameters, &residual);
    const double target =
        std::max(32.0 * std::numeric_limits<double>::epsilon(),
                 options.objective_tolerance) *
        std::max(1.0,
                 std::abs(minimax_error(alternant.front(), parameters)));
    final_residual = current_residual;
    final_target = target;
    if(current_residual <= target) {
      parameters.alternating_error =
          std::abs(minimax_error(alternant.front(), parameters));
      return parameters;
    }

    linalg::Matrix jacobian{dimension, dimension};
    linalg::Matrix right_hand_side{dimension, 1};
    for(std::size_t row = 0; row < dimension; ++row) {
      const double left_denominator = alternant[row];
      const double right_denominator = alternant[row + 1];
      right_hand_side(row, 0) = -residual[row];
      for(std::size_t point = 0; point < points; ++point) {
        const double left_decay = safe_decay(
            left_denominator, parameters.nodes[point]);
        const double right_decay = safe_decay(
            right_denominator, parameters.nodes[point]);
        jacobian(row, point) = left_decay + right_decay;
        jacobian(row, points + point) =
            -parameters.weights[point] * parameters.nodes[point] *
            (left_denominator * left_decay +
             right_denominator * right_decay);
      }
    }
    const linalg::LeastSquaresResult step =
        linalg::solve_least_squares_svd(
            std::move(jacobian), right_hand_side, 1.0e-16);
    if(step.numerical_rank != dimension) {
      throw std::runtime_error(
          "minimax alternation Jacobian is rank deficient");
    }
    if(!step.singular_values.empty()) {
      const auto [minimum, maximum] = std::minmax_element(
          step.singular_values.begin(), step.singular_values.end());
      if(*minimum > 0.0 && std::isfinite(*maximum)) {
        parameters.jacobian_condition_number = *maximum / *minimum;
      }
    }

    bool accepted = false;
    double best_residual = current_residual;
    MinimaxParameters best = parameters;
    for(std::size_t line_search = 0; line_search < 30; ++line_search) {
      const double scale = std::ldexp(1.0, -static_cast<int>(line_search));
      MinimaxParameters candidate = parameters;
      bool valid = true;
      for(std::size_t point = 0; point < points; ++point) {
        candidate.weights[point] += scale * step.solution(point, 0);
        const double logarithmic_node =
            std::log(parameters.nodes[point]) +
            scale * step.solution(points + point, 0);
        if(!std::isfinite(candidate.weights[point]) ||
           !std::isfinite(logarithmic_node) ||
           logarithmic_node < -700.0 || logarithmic_node > 700.0) {
          valid = false;
          break;
        }
        candidate.nodes[point] = std::exp(logarithmic_node);
      }
      if(!valid) {
        continue;
      }
      sort_minimax_node_weight_pairs(candidate);
      for(std::size_t point = 0; point < points; ++point) {
        if(candidate.nodes[point] <= 0.0 ||
           (point != 0 &&
            candidate.nodes[point] <= candidate.nodes[point - 1])) {
          valid = false;
          break;
        }
      }
      if(!valid) {
        continue;
      }
      const double candidate_residual = adjacent_alternation_residual(
          alternant, candidate, nullptr);
      if(candidate_residual < best_residual) {
        best_residual = candidate_residual;
        best = std::move(candidate);
      }
      if(candidate_residual <= 0.9 * current_residual) {
        parameters = std::move(best);
        accepted = true;
        break;
      }
    }
    if(!accepted) {
      if(best_residual < current_residual) {
        parameters = std::move(best);
      } else {
        throw std::runtime_error(
            "minimax alternation Newton line search did not improve; "
            "residual=" + std::to_string(current_residual) +
            " target=" + std::to_string(target));
      }
    }
  }
  throw std::runtime_error(
      "minimax alternation Newton solver did not converge; residual=" +
      std::to_string(final_residual) +
      " target=" + std::to_string(final_target));
}

[[nodiscard]] double bisect_minimax_zero(
    double lower, double upper, const MinimaxParameters& parameters) {
  double lower_value = minimax_error(lower, parameters);
  double upper_value = minimax_error(upper, parameters);
  if(lower_value == 0.0) {
    return lower;
  }
  if(upper_value == 0.0) {
    return upper;
  }
  if(std::signbit(lower_value) == std::signbit(upper_value)) {
    throw std::runtime_error(
        "minimax alternant does not bracket an error zero");
  }
  for(std::size_t iteration = 0; iteration < 100; ++iteration) {
    const double midpoint = std::exp(
        0.5 * (std::log(lower) + std::log(upper)));
    const double midpoint_value = minimax_error(midpoint, parameters);
    if(midpoint_value == 0.0 ||
       std::abs(std::log(upper / lower)) <= 2.0e-14) {
      return midpoint;
    }
    if(std::signbit(lower_value) == std::signbit(midpoint_value)) {
      lower = midpoint;
      lower_value = midpoint_value;
    } else {
      upper = midpoint;
      upper_value = midpoint_value;
    }
  }
  return std::exp(0.5 * (std::log(lower) + std::log(upper)));
}

[[nodiscard]] double maximize_minimax_absolute_error(
    double lower, double upper, const MinimaxParameters& parameters) {
  if(lower == upper) {
    return lower;
  }
  constexpr double golden = 0.6180339887498948482;
  double left = std::log(lower);
  double right = std::log(upper);
  double x1 = right - golden * (right - left);
  double x2 = left + golden * (right - left);
  auto objective = [&](double logarithmic_denominator) {
    return std::abs(minimax_error(std::exp(logarithmic_denominator),
                                  parameters));
  };
  double f1 = objective(x1);
  double f2 = objective(x2);
  for(std::size_t iteration = 0; iteration < 100; ++iteration) {
    if(f1 < f2) {
      left = x1;
      x1 = x2;
      f1 = f2;
      x2 = left + golden * (right - left);
      f2 = objective(x2);
    } else {
      right = x2;
      x2 = x1;
      f2 = f1;
      x1 = right - golden * (right - left);
      f1 = objective(x1);
    }
  }
  std::array<double, 3> candidates{
      lower, std::exp(0.5 * (left + right)), upper};
  return *std::max_element(
      candidates.begin(), candidates.end(),
      [&](double lhs, double rhs) {
        return std::abs(minimax_error(lhs, parameters)) <
               std::abs(minimax_error(rhs, parameters));
      });
}

[[nodiscard]] LaplaceFitResult minimax_quadrature(
    const DenominatorHistogram& histogram,
    const LaplaceFitOptions& options) {
  const std::size_t points = options.number_of_points;
  const double denominator_minimum = histogram.denominator_minimum;
  const double denominator_maximum = histogram.denominator_maximum;
  const double ratio = denominator_maximum / denominator_minimum;
  if(!std::isfinite(ratio) || ratio <= 1.0) {
    throw std::invalid_argument(
        "minimax quadrature requires a finite nondegenerate interval");
  }
  if(points < 5 || points > 7) {
    throw std::invalid_argument(
        "the current minimax implementation supports 5, 6, or 7 "
        "quadrature points");
  }

  MinimaxParameters parameters =
      points == 5 ? minimax_seed(points, ratio)
                  : enclosing_minimax_seed(points, ratio);
  std::vector<double> history;
  std::vector<double> condition_history;
  std::size_t completed_iterations = 0;
  if(points == 5) {
    const std::size_t alternant_size = 2 * points + 1;
    std::vector<double> alternant =
        initial_minimax_alternant(ratio, points, parameters);
    const std::size_t maximum_remez_iterations =
        std::min<std::size_t>(options.maximum_optimizer_iterations, 100);
    for(; completed_iterations < maximum_remez_iterations;
        ++completed_iterations) {
      parameters = solve_minimax_alternation(
          alternant, std::move(parameters), options);
      std::vector<double> roots(alternant_size - 1);
      for(std::size_t interval = 0; interval < roots.size(); ++interval) {
        roots[interval] = bisect_minimax_zero(
            alternant[interval], alternant[interval + 1], parameters);
      }
      std::vector<double> next_alternant(alternant_size);
      for(std::size_t interval = 0; interval < alternant_size; ++interval) {
        const double lower = interval == 0 ? 1.0 : roots[interval - 1];
        const double upper =
            interval == roots.size() ? ratio : roots[interval];
        next_alternant[interval] = maximize_minimax_absolute_error(
            lower, upper, parameters);
      }
      next_alternant.front() = 1.0;
      next_alternant.back() = ratio;

      double minimum_amplitude = std::numeric_limits<double>::infinity();
      double maximum_amplitude = 0.0;
      for(const double denominator : next_alternant) {
        const double amplitude = std::abs(minimax_error(
            denominator, parameters));
        minimum_amplitude = std::min(minimum_amplitude, amplitude);
        maximum_amplitude = std::max(maximum_amplitude, amplitude);
      }
      const double maximum_squared_error =
          maximum_amplitude * maximum_amplitude;
      const double recorded_objective =
          history.empty()
              ? maximum_squared_error
              : std::min(history.back(), maximum_squared_error);
      history.push_back(recorded_objective);
      condition_history.push_back(
          parameters.jacobian_condition_number);
      alternant = std::move(next_alternant);
      const double tolerance = options.objective_tolerance *
                               std::max(1.0, maximum_amplitude);
      if(maximum_amplitude - minimum_amplitude <= tolerance) {
        ++completed_iterations;
        parameters = solve_minimax_alternation(
            alternant, std::move(parameters), options);
        break;
      }
    }
    if(completed_iterations == maximum_remez_iterations) {
      throw std::runtime_error(
          "minimax Remez optimizer did not converge");
    }
  }

  for(std::size_t point = 0; point < points; ++point) {
    parameters.nodes[point] /= denominator_minimum;
    parameters.weights[point] /= denominator_minimum;
    if(!std::isfinite(parameters.nodes[point]) ||
       parameters.nodes[point] <= 0.0 ||
       !std::isfinite(parameters.weights[point]) ||
       parameters.weights[point] <= 0.0) {
      throw std::runtime_error(
          "minimax optimization produced a nonpositive node or weight");
    }
  }
  sort_minimax_node_weight_pairs(parameters);

  std::vector<double> validation_grid(options.validation_grid_size);
  const double log_minimum = std::log(denominator_minimum);
  const double log_maximum = std::log(denominator_maximum);
  for(std::size_t point = 0; point < validation_grid.size(); ++point) {
    const double fraction = static_cast<double>(point) /
                            static_cast<double>(validation_grid.size() - 1);
    validation_grid[point] =
        std::exp(log_minimum + fraction * (log_maximum - log_minimum));
  }
  validation_grid.front() = denominator_minimum;
  validation_grid.back() = denominator_maximum;

  long double weighted_objective = 0.0L;
  if(!histogram.midpoints.empty()) {
    for(std::size_t bin = 0; bin < histogram.midpoints.size(); ++bin) {
      const double error =
          evaluate_laplace_reciprocal(
              histogram.midpoints[bin], parameters.nodes,
              parameters.weights) -
          1.0 / histogram.midpoints[bin];
      weighted_objective +=
          static_cast<long double>(histogram.frequencies[bin]) *
          static_cast<long double>(error) *
          static_cast<long double>(error);
    }
  } else {
    for(const double denominator : validation_grid) {
      const double error = evaluate_laplace_reciprocal(
                               denominator, parameters.nodes,
                               parameters.weights) -
                           1.0 / denominator;
      weighted_objective += static_cast<long double>(error) *
                            static_cast<long double>(error) /
                            static_cast<long double>(validation_grid.size());
    }
  }
  const double objective = static_cast<double>(weighted_objective);
  if(history.empty() || history.back() != objective) {
    if(!history.empty() && objective > history.back() +
                               64.0 * std::numeric_limits<double>::epsilon() *
                                   std::max(1.0, history.back())) {
      throw std::runtime_error(
          "minimax weighted objective exceeds the final uniform error bound");
    }
    history.push_back(objective);
    condition_history.push_back(parameters.jacobian_condition_number);
  }
  double histogram_maximum_absolute = 0.0;
  double histogram_maximum_relative = 0.0;
  error_diagnostics(
      histogram.midpoints.empty() ? validation_grid : histogram.midpoints,
      parameters.nodes, parameters.weights,
      histogram_maximum_absolute, histogram_maximum_relative);
  double validation_maximum_absolute = 0.0;
  double validation_maximum_relative = 0.0;
  error_diagnostics(validation_grid, parameters.nodes, parameters.weights,
                    validation_maximum_absolute,
                    validation_maximum_relative);

  return LaplaceFitResult{
      .histogram = histogram,
      .nodes = std::move(parameters.nodes),
      .weights = std::move(parameters.weights),
      .optimization_history = std::move(history),
      .condition_number_history = std::move(condition_history),
      .optimizer_iterations = completed_iterations,
      .weight_numerical_rank = points,
      .weighted_objective = objective,
      .weighted_rmse = std::sqrt(objective),
      .histogram_maximum_absolute_error = histogram_maximum_absolute,
      .histogram_maximum_relative_error = histogram_maximum_relative,
      .validation_maximum_absolute_error = validation_maximum_absolute,
      .validation_maximum_relative_error = validation_maximum_relative,
  };
}

[[nodiscard]] WeightFit solve_weights(
    const DenominatorHistogram& histogram,
    std::vector<double> nodes,
    const LaplaceFitOptions& options) {
  std::sort(nodes.begin(), nodes.end());
  const std::size_t rows = histogram.midpoints.size();
  const std::size_t columns = nodes.size();
  linalg::Matrix design{rows, columns};
  linalg::Matrix right_hand_side{rows, 1};
  for(std::size_t row = 0; row < rows; ++row) {
    const double root_weight = std::sqrt(histogram.frequencies[row]);
    const double denominator = histogram.midpoints[row];
    right_hand_side(row, 0) = root_weight / denominator;
    for(std::size_t column = 0; column < columns; ++column) {
      design(row, column) =
          root_weight * safe_decay(denominator, nodes[column]);
    }
  }

  linalg::Matrix gram{columns, columns};
  linalg::Matrix normal_rhs{columns, 1};
  for(std::size_t left = 0; left < columns; ++left) {
    for(std::size_t row = 0; row < rows; ++row) {
      normal_rhs(left, 0) +=
          design(row, left) * right_hand_side(row, 0);
    }
    for(std::size_t right = 0; right <= left; ++right) {
      double value = 0.0;
      for(std::size_t row = 0; row < rows; ++row) {
        value += design(row, left) * design(row, right);
      }
      gram(left, right) = value;
      gram(right, left) = value;
    }
  }
  const linalg::SymmetricEigendecomposition spectrum =
      linalg::diagonalize_symmetric(gram, 2.0e-13);
  const double largest_eigenvalue = spectrum.eigenvalues.back();
  const double smallest_eigenvalue = spectrum.eigenvalues.front();
  double condition_number = std::numeric_limits<double>::infinity();
  if(largest_eigenvalue > 0.0 && smallest_eigenvalue > 0.0) {
    condition_number = largest_eigenvalue / smallest_eigenvalue;
  }

  linalg::Matrix solution;
  std::size_t numerical_rank = columns;
  if(std::isfinite(condition_number) &&
     condition_number <=
         options.normal_equation_condition_threshold) {
    solution = linalg::solve_symmetric_positive_definite(
        std::move(gram), std::move(normal_rhs));
  } else {
    const linalg::LeastSquaresResult svd =
        linalg::solve_least_squares_svd(
            std::move(design), right_hand_side,
            options.relative_singular_value_cutoff);
    solution = svd.solution;
    numerical_rank = svd.numerical_rank;
    if(!svd.singular_values.empty() &&
       svd.singular_values.front() > 0.0 &&
       svd.singular_values.back() > 0.0) {
      const double ratio = svd.singular_values.front() /
                           svd.singular_values.back();
      condition_number = ratio * ratio;
    }
  }

  std::vector<double> weights(columns);
  for(std::size_t column = 0; column < columns; ++column) {
    weights[column] = solution(column, 0);
  }
  long double objective = 0.0L;
  for(std::size_t row = 0; row < rows; ++row) {
    const double denominator = histogram.midpoints[row];
    const double approximation =
        evaluate_laplace_reciprocal(denominator, nodes, weights);
    const double residual = 1.0 / denominator - approximation;
    objective +=
        static_cast<long double>(histogram.frequencies[row]) *
        static_cast<long double>(residual) *
        static_cast<long double>(residual);
  }
  const double objective_value = static_cast<double>(objective);
  if(!std::isfinite(objective_value) || objective_value < 0.0) {
    throw std::runtime_error(
        "Laplace weighted fit produced an invalid objective");
  }
  return WeightFit{
      .nodes = std::move(nodes),
      .weights = std::move(weights),
      .objective = objective_value,
      .condition_number = condition_number,
      .numerical_rank = numerical_rank,
  };
}

struct SimplexVertex {
  std::vector<double> logarithmic_nodes;
  WeightFit fit;
};

[[nodiscard]] bool vertex_less(const SimplexVertex& left,
                               const SimplexVertex& right) {
  if(left.fit.objective != right.fit.objective) {
    return left.fit.objective < right.fit.objective;
  }
  return left.logarithmic_nodes < right.logarithmic_nodes;
}

void normalize_logarithmic_nodes(std::vector<double>& values,
                                 double lower, double upper) {
  for(double& value : values) {
    value = std::clamp(value, lower, upper);
  }
  std::sort(values.begin(), values.end());
}

[[nodiscard]] std::vector<double> physical_nodes(
    const std::vector<double>& logarithmic_nodes) {
  std::vector<double> result;
  result.reserve(logarithmic_nodes.size());
  for(const double value : logarithmic_nodes) {
    const double node = std::exp(value);
    if(!std::isfinite(node) || node <= 0.0) {
      throw std::runtime_error(
          "Laplace optimizer produced an invalid positive node");
    }
    result.push_back(node);
  }
  return result;
}

[[nodiscard]] SimplexVertex make_vertex(
    std::vector<double> logarithmic_nodes, double lower, double upper,
    const DenominatorHistogram& histogram,
    const LaplaceFitOptions& options) {
  normalize_logarithmic_nodes(logarithmic_nodes, lower, upper);
  return SimplexVertex{
      .logarithmic_nodes = logarithmic_nodes,
      .fit = solve_weights(
          histogram, physical_nodes(logarithmic_nodes), options),
  };
}

[[nodiscard]] std::vector<double> affine_trial(
    const std::vector<double>& centroid,
    const std::vector<double>& reference, double factor) {
  std::vector<double> result(centroid.size());
  for(std::size_t index = 0; index < result.size(); ++index) {
    result[index] =
        centroid[index] + factor * (centroid[index] - reference[index]);
  }
  return result;
}

void error_diagnostics(const std::vector<double>& denominators,
                       const std::vector<double>& nodes,
                       const std::vector<double>& weights,
                       double& maximum_absolute,
                       double& maximum_relative) {
  maximum_absolute = 0.0;
  maximum_relative = 0.0;
  for(const double denominator : denominators) {
    const double exact = 1.0 / denominator;
    const double approximate =
        evaluate_laplace_reciprocal(denominator, nodes, weights);
    const double absolute = std::abs(exact - approximate);
    maximum_absolute = std::max(maximum_absolute, absolute);
    maximum_relative =
        std::max(maximum_relative, absolute / exact);
  }
}

[[nodiscard]] LaplaceFitResult gauss_laguerre_quadrature(
    const DenominatorHistogram& histogram,
    const LaplaceFitOptions& options) {
  const std::size_t points = options.number_of_points;
  linalg::Matrix jacobi{points, points};
  for(std::size_t point = 0; point < points; ++point) {
    jacobi(point, point) = 2.0 * static_cast<double>(point) + 1.0;
    if(point != 0) {
      const double off_diagonal = static_cast<double>(point);
      jacobi(point - 1, point) = off_diagonal;
      jacobi(point, point - 1) = off_diagonal;
    }
  }
  const auto eigensystem = linalg::diagonalize_symmetric(jacobi);
  std::vector<double> nodes = eigensystem.eigenvalues;
  std::vector<double> weights(points);
  for(std::size_t point = 0; point < points; ++point) {
    const double first_component = eigensystem.eigenvectors(0, point);
    const double raw_weight = first_component * first_component;
    if(!std::isfinite(nodes[point]) || nodes[point] <= 0.0 ||
       !std::isfinite(raw_weight) || raw_weight <= 0.0) {
      throw std::runtime_error(
          "Gauss-Laguerre eigensystem contains an invalid node or weight");
    }

    // Gauss-Laguerre supplies w_k for integral exp(-t) f(t) dt. The
    // Laplace kernels consume W_k exp(-D t_k), so W_k = w_k exp(t_k).
    const double logarithmic_effective_weight =
        std::log(raw_weight) + nodes[point];
    if(!std::isfinite(logarithmic_effective_weight) ||
       logarithmic_effective_weight >
           std::log(std::numeric_limits<double>::max())) {
      throw std::overflow_error(
          "Gauss-Laguerre effective weight exceeds double precision");
    }
    weights[point] = std::exp(logarithmic_effective_weight);
  }

  std::vector<double> validation_grid(options.validation_grid_size);
  const double log_minimum = std::log(histogram.denominator_minimum);
  const double log_maximum = std::log(histogram.denominator_maximum);
  for(std::size_t point = 0; point < options.validation_grid_size; ++point) {
    const double fraction =
        static_cast<double>(point) /
        static_cast<double>(options.validation_grid_size - 1);
    validation_grid[point] =
        std::exp(log_minimum + fraction * (log_maximum - log_minimum));
  }
  validation_grid.front() = histogram.denominator_minimum;
  validation_grid.back() = histogram.denominator_maximum;

  long double weighted_objective = 0.0L;
  for(const double denominator : validation_grid) {
    const double exact = 1.0 / denominator;
    const double approximate = evaluate_laplace_reciprocal(
        denominator, nodes, weights);
    const long double residual =
        static_cast<long double>(approximate - exact);
    weighted_objective += residual * residual;
  }
  const double objective = static_cast<double>(
      weighted_objective /
      static_cast<long double>(validation_grid.size()));

  double histogram_maximum_absolute = 0.0;
  double histogram_maximum_relative = 0.0;
  error_diagnostics(validation_grid, nodes, weights,
                    histogram_maximum_absolute,
                    histogram_maximum_relative);
  double validation_maximum_absolute = 0.0;
  double validation_maximum_relative = 0.0;
  error_diagnostics(validation_grid, nodes, weights,
                    validation_maximum_absolute,
                    validation_maximum_relative);

  return LaplaceFitResult{
      .histogram = histogram,
      .nodes = std::move(nodes),
      .weights = std::move(weights),
      .optimization_history = {},
      .condition_number_history = {},
      .optimizer_iterations = 0,
      .weight_numerical_rank = points,
      .weighted_objective = objective,
      .weighted_rmse = std::sqrt(objective),
      .histogram_maximum_absolute_error = histogram_maximum_absolute,
      .histogram_maximum_relative_error = histogram_maximum_relative,
      .validation_maximum_absolute_error = validation_maximum_absolute,
      .validation_maximum_relative_error = validation_maximum_relative,
  };
}

}  // namespace

DenominatorHistogram build_denominator_histogram(
    const std::vector<double>& orbital_energies,
    const std::vector<std::size_t>& active_occupied_indices,
    const std::vector<std::size_t>& virtual_indices,
    const LaplaceFitOptions& options) {
  validate_options(options);
  validate_orbital_space(orbital_energies, active_occupied_indices,
                         virtual_indices);

  double occupied_minimum = std::numeric_limits<double>::infinity();
  double occupied_maximum = -std::numeric_limits<double>::infinity();
  for(const std::size_t orbital : active_occupied_indices) {
    occupied_minimum =
        std::min(occupied_minimum, orbital_energies[orbital]);
    occupied_maximum =
        std::max(occupied_maximum, orbital_energies[orbital]);
  }
  double virtual_minimum = std::numeric_limits<double>::infinity();
  double virtual_maximum = -std::numeric_limits<double>::infinity();
  for(const std::size_t orbital : virtual_indices) {
    virtual_minimum =
        std::min(virtual_minimum, orbital_energies[orbital]);
    virtual_maximum =
        std::max(virtual_maximum, orbital_energies[orbital]);
  }
  const double denominator_minimum =
      2.0 * virtual_minimum - 2.0 * occupied_maximum;
  const double denominator_maximum =
      2.0 * virtual_maximum - 2.0 * occupied_minimum;
  if(!std::isfinite(denominator_minimum) ||
     !std::isfinite(denominator_maximum) ||
     denominator_minimum <= 0.0 ||
     denominator_maximum <= denominator_minimum) {
    throw std::runtime_error(
        "Laplace denominator interval is nonpositive or degenerate");
  }

  std::vector<double> gaps;
  if(active_occupied_indices.size() >
     std::numeric_limits<std::size_t>::max() / virtual_indices.size()) {
    throw std::overflow_error("occupied-virtual gap count overflows size_t");
  }
  gaps.reserve(active_occupied_indices.size() * virtual_indices.size());
  for(const std::size_t occupied : active_occupied_indices) {
    for(const std::size_t virtual_orbital : virtual_indices) {
      const double gap =
          orbital_energies[virtual_orbital] -
          orbital_energies[occupied];
      if(!std::isfinite(gap) || gap <= 0.0) {
        throw std::runtime_error(
            "Laplace occupied-virtual gap is not strictly positive");
      }
      gaps.push_back(gap);
    }
  }
  const std::uint64_t expected_total = checked_square_count(gaps.size());
  const double logarithmic_minimum = std::log(denominator_minimum);
  const double logarithmic_maximum = std::log(denominator_maximum);
  const double logarithmic_width =
      (logarithmic_maximum - logarithmic_minimum) /
      static_cast<double>(options.histogram_bins);
  std::vector<double> edges(options.histogram_bins + 1);
  std::vector<double> midpoints(options.histogram_bins);
  for(std::size_t bin = 0; bin <= options.histogram_bins; ++bin) {
    edges[bin] = std::exp(
        logarithmic_minimum +
        logarithmic_width * static_cast<double>(bin));
  }
  edges.front() = denominator_minimum;
  edges.back() = denominator_maximum;
  for(std::size_t bin = 0; bin < options.histogram_bins; ++bin) {
    midpoints[bin] = std::exp(
        logarithmic_minimum +
        logarithmic_width * (static_cast<double>(bin) + 0.5));
  }

  const std::size_t block_count =
      1 + (gaps.size() - 1) / options.gap_block_size;
  std::vector<std::size_t> owned_blocks;
  for(std::size_t block = options.mpi_rank; block < block_count;
      block += options.mpi_ranks) {
    owned_blocks.push_back(block);
  }
  std::vector<std::vector<std::uint64_t>> thread_counts(
      options.threads,
      std::vector<std::uint64_t>(options.histogram_bins, 0));
  std::vector<int> thread_failures(options.threads, 0);

#ifdef MODERNQC_HAS_OPENMP
#pragma omp parallel num_threads(static_cast<int>(options.threads))
  {
    const std::size_t thread =
        static_cast<std::size_t>(omp_get_thread_num());
    const std::size_t active_threads =
        static_cast<std::size_t>(omp_get_num_threads());
    for(std::size_t owned = thread; owned < owned_blocks.size();
        owned += active_threads) {
#else
  {
    const std::size_t thread = 0;
    for(std::size_t owned = 0; owned < owned_blocks.size(); ++owned) {
#endif
      const std::size_t first =
          owned_blocks[owned] * options.gap_block_size;
      const std::size_t last =
          std::min(first + options.gap_block_size, gaps.size());
      for(std::size_t left = first; left < last; ++left) {
        for(const double right_gap : gaps) {
          const double denominator = gaps[left] + right_gap;
          if(!std::isfinite(denominator) ||
             denominator < denominator_minimum ||
             denominator > denominator_maximum) {
            thread_failures[thread] = 1;
            continue;
          }
          const double position =
              (std::log(denominator) - logarithmic_minimum) /
              logarithmic_width;
          const double bounded_position =
              std::clamp(position, 0.0,
                         static_cast<double>(options.histogram_bins));
          std::size_t bin =
              bounded_position >=
                      static_cast<double>(options.histogram_bins)
                  ? options.histogram_bins - 1
                  : static_cast<std::size_t>(bounded_position);
          bin = std::min(bin, options.histogram_bins - 1);
          if(thread_counts[thread][bin] ==
             std::numeric_limits<std::uint64_t>::max()) {
            thread_failures[thread] = 2;
            continue;
          }
          ++thread_counts[thread][bin];
        }
      }
    }
  }

  for(const int failure : thread_failures) {
    if(failure == 1) {
      throw std::runtime_error(
          "Laplace denominator lies outside its analytic interval");
    }
    if(failure == 2) {
      throw std::overflow_error(
          "Laplace histogram bin count overflows uint64");
    }
  }
  std::vector<std::uint64_t> counts(options.histogram_bins, 0);
  for(std::size_t thread = 0; thread < options.threads; ++thread) {
    for(std::size_t bin = 0; bin < options.histogram_bins; ++bin) {
      if(counts[bin] >
         std::numeric_limits<std::uint64_t>::max() -
             thread_counts[thread][bin]) {
        throw std::overflow_error(
            "Laplace histogram reduction overflows uint64");
      }
      counts[bin] += thread_counts[thread][bin];
    }
  }
  reduce_histogram_across_ranks(counts, options);
  const std::uint64_t observed_total =
      std::accumulate(counts.begin(), counts.end(), std::uint64_t{0});
  if(observed_total != expected_total) {
    throw std::runtime_error(
        "Laplace histogram count does not equal the ordered denominator "
        "count");
  }
  std::vector<double> frequencies(options.histogram_bins);
  double frequency_sum = 0.0;
  for(std::size_t bin = 0; bin < options.histogram_bins; ++bin) {
    frequencies[bin] =
        static_cast<double>(counts[bin]) /
        static_cast<double>(observed_total);
    frequency_sum += frequencies[bin];
  }
  const double frequency_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() *
      static_cast<double>(options.histogram_bins);
  if(!std::isfinite(frequency_sum) ||
     std::abs(frequency_sum - 1.0) > frequency_tolerance) {
    throw std::runtime_error(
        "Laplace histogram frequencies do not sum to one");
  }
  return DenominatorHistogram{
      .denominator_minimum = denominator_minimum,
      .denominator_maximum = denominator_maximum,
      .total_ordered_denominators = observed_total,
      .edges = std::move(edges),
      .midpoints = std::move(midpoints),
      .counts = std::move(counts),
      .frequencies = std::move(frequencies),
  };
}

LaplaceFitResult fit_laplace_quadrature(
    const DenominatorHistogram& histogram,
    const LaplaceFitOptions& options) {
  validate_options(options);
  if(options.scheme == LaplaceQuadratureScheme::gauss_laguerre) {
    if(!std::isfinite(histogram.denominator_minimum) ||
       !std::isfinite(histogram.denominator_maximum) ||
       histogram.denominator_minimum <= 0.0 ||
       histogram.denominator_maximum <= histogram.denominator_minimum ||
       histogram.total_ordered_denominators == 0) {
      throw std::invalid_argument(
          "Laplace denominator interval is incompatible with "
          "Gauss-Laguerre quadrature");
    }
    return gauss_laguerre_quadrature(histogram, options);
  }
  if(histogram.midpoints.size() != options.histogram_bins ||
     histogram.counts.size() != options.histogram_bins ||
     histogram.frequencies.size() != options.histogram_bins ||
     histogram.edges.size() != options.histogram_bins + 1 ||
     !std::isfinite(histogram.denominator_minimum) ||
     !std::isfinite(histogram.denominator_maximum) ||
     histogram.denominator_minimum <= 0.0 ||
     histogram.denominator_maximum <= histogram.denominator_minimum ||
     histogram.total_ordered_denominators == 0) {
    throw std::invalid_argument(
        "Laplace denominator histogram is incompatible with fit options");
  }
  std::uint64_t count_sum = 0;
  double frequency_sum = 0.0;
  for(std::size_t bin = 0; bin < options.histogram_bins; ++bin) {
    if(!std::isfinite(histogram.midpoints[bin]) ||
       histogram.midpoints[bin] <= 0.0 ||
       !std::isfinite(histogram.frequencies[bin]) ||
       histogram.frequencies[bin] < 0.0 ||
       count_sum > std::numeric_limits<std::uint64_t>::max() -
                       histogram.counts[bin]) {
      throw std::invalid_argument(
          "Laplace denominator histogram contains invalid data");
    }
    count_sum += histogram.counts[bin];
    frequency_sum += histogram.frequencies[bin];
  }
  if(count_sum != histogram.total_ordered_denominators ||
     std::abs(frequency_sum - 1.0) >
         64.0 * std::numeric_limits<double>::epsilon() *
             static_cast<double>(options.histogram_bins)) {
    throw std::invalid_argument(
        "Laplace denominator histogram normalization is invalid");
  }
  if(options.scheme == LaplaceQuadratureScheme::minimax) {
    return minimax_quadrature(histogram, options);
  }

  const double initial_minimum =
      0.05 / histogram.denominator_maximum;
  const double initial_maximum =
      20.0 / histogram.denominator_minimum;
  const double logarithmic_initial_minimum = std::log(initial_minimum);
  const double logarithmic_initial_maximum = std::log(initial_maximum);
  const double logarithmic_lower =
      std::log(1.0e-4 / histogram.denominator_maximum);
  const double logarithmic_upper =
      std::log(1.0e4 / histogram.denominator_minimum);
  std::vector<double> initial(options.number_of_points);
  for(std::size_t point = 0; point < options.number_of_points; ++point) {
    const double fraction =
        options.number_of_points == 1
            ? 0.5
            : static_cast<double>(point) /
                  static_cast<double>(options.number_of_points - 1);
    initial[point] =
        logarithmic_initial_minimum +
        fraction *
            (logarithmic_initial_maximum -
             logarithmic_initial_minimum);
  }

  std::vector<SimplexVertex> simplex;
  simplex.reserve(options.number_of_points + 1);
  simplex.push_back(make_vertex(initial, logarithmic_lower,
                                logarithmic_upper, histogram, options));
  for(std::size_t point = 0; point < options.number_of_points; ++point) {
    std::vector<double> vertex = initial;
    vertex[point] += options.initial_simplex_step;
    simplex.push_back(make_vertex(
        std::move(vertex), logarithmic_lower, logarithmic_upper,
        histogram, options));
  }

  std::vector<double> objective_history;
  std::vector<double> condition_history;
  std::size_t completed_iterations = 0;
  for(; completed_iterations < options.maximum_optimizer_iterations;
      ++completed_iterations) {
    std::sort(simplex.begin(), simplex.end(), vertex_less);
    objective_history.push_back(simplex.front().fit.objective);
    condition_history.push_back(
        simplex.front().fit.condition_number);
    double objective_spread = 0.0;
    double node_spread = 0.0;
    for(std::size_t vertex = 1; vertex < simplex.size(); ++vertex) {
      objective_spread =
          std::max(objective_spread,
                   std::abs(simplex[vertex].fit.objective -
                            simplex.front().fit.objective));
      for(std::size_t point = 0; point < options.number_of_points; ++point) {
        node_spread =
            std::max(node_spread,
                     std::abs(simplex[vertex].logarithmic_nodes[point] -
                              simplex.front().logarithmic_nodes[point]));
      }
    }
    const double scaled_objective_tolerance =
        options.objective_tolerance *
        std::max(1.0, simplex.front().fit.objective);
    if(objective_spread <= scaled_objective_tolerance &&
       node_spread <= options.logarithmic_node_tolerance) {
      break;
    }

    std::vector<double> centroid(options.number_of_points, 0.0);
    for(std::size_t vertex = 0; vertex < options.number_of_points; ++vertex) {
      for(std::size_t point = 0; point < options.number_of_points; ++point) {
        centroid[point] += simplex[vertex].logarithmic_nodes[point];
      }
    }
    for(double& value : centroid) {
      value /= static_cast<double>(options.number_of_points);
    }
    SimplexVertex reflected = make_vertex(
        affine_trial(centroid, simplex.back().logarithmic_nodes, 1.0),
        logarithmic_lower, logarithmic_upper, histogram, options);
    if(reflected.fit.objective < simplex.front().fit.objective) {
      SimplexVertex expanded = make_vertex(
          affine_trial(centroid, simplex.back().logarithmic_nodes, 2.0),
          logarithmic_lower, logarithmic_upper, histogram, options);
      simplex.back() =
          expanded.fit.objective < reflected.fit.objective
              ? std::move(expanded)
              : std::move(reflected);
      continue;
    }
    if(reflected.fit.objective <
       simplex[options.number_of_points - 1].fit.objective) {
      simplex.back() = std::move(reflected);
      continue;
    }

    const bool outside =
        reflected.fit.objective < simplex.back().fit.objective;
    const std::vector<double>& contraction_reference =
        outside ? reflected.logarithmic_nodes
                : simplex.back().logarithmic_nodes;
    std::vector<double> contracted_values(options.number_of_points);
    for(std::size_t point = 0; point < options.number_of_points; ++point) {
      contracted_values[point] =
          centroid[point] +
          0.5 * (contraction_reference[point] - centroid[point]);
    }
    SimplexVertex contracted = make_vertex(
        std::move(contracted_values), logarithmic_lower,
        logarithmic_upper, histogram, options);
    const double contraction_target =
        outside ? reflected.fit.objective : simplex.back().fit.objective;
    if(contracted.fit.objective < contraction_target) {
      simplex.back() = std::move(contracted);
      continue;
    }

    const std::vector<double> best_nodes =
        simplex.front().logarithmic_nodes;
    for(std::size_t vertex = 1; vertex < simplex.size(); ++vertex) {
      std::vector<double> shrunk(options.number_of_points);
      for(std::size_t point = 0; point < options.number_of_points; ++point) {
        shrunk[point] =
            best_nodes[point] +
            0.5 * (simplex[vertex].logarithmic_nodes[point] -
                   best_nodes[point]);
      }
      simplex[vertex] = make_vertex(
          std::move(shrunk), logarithmic_lower, logarithmic_upper,
          histogram, options);
    }
  }
  if(completed_iterations == options.maximum_optimizer_iterations) {
    throw std::runtime_error(
        "Laplace logarithmic-node Nelder-Mead optimizer did not converge");
  }
  std::sort(simplex.begin(), simplex.end(), vertex_less);
  const WeightFit& final_fit = simplex.front().fit;
  if(objective_history.empty() ||
     objective_history.back() != final_fit.objective) {
    objective_history.push_back(final_fit.objective);
    condition_history.push_back(final_fit.condition_number);
  }

  double histogram_maximum_absolute = 0.0;
  double histogram_maximum_relative = 0.0;
  error_diagnostics(
      histogram.midpoints, final_fit.nodes, final_fit.weights,
      histogram_maximum_absolute, histogram_maximum_relative);
  std::vector<double> validation_grid(options.validation_grid_size);
  const double log_minimum = std::log(histogram.denominator_minimum);
  const double log_maximum = std::log(histogram.denominator_maximum);
  for(std::size_t point = 0; point < options.validation_grid_size; ++point) {
    const double fraction =
        static_cast<double>(point) /
        static_cast<double>(options.validation_grid_size - 1);
    validation_grid[point] =
        std::exp(log_minimum + fraction * (log_maximum - log_minimum));
  }
  validation_grid.front() = histogram.denominator_minimum;
  validation_grid.back() = histogram.denominator_maximum;
  double validation_maximum_absolute = 0.0;
  double validation_maximum_relative = 0.0;
  error_diagnostics(
      validation_grid, final_fit.nodes, final_fit.weights,
      validation_maximum_absolute, validation_maximum_relative);

  return LaplaceFitResult{
      .histogram = histogram,
      .nodes = final_fit.nodes,
      .weights = final_fit.weights,
      .optimization_history = std::move(objective_history),
      .condition_number_history = std::move(condition_history),
      .optimizer_iterations = completed_iterations,
      .weight_numerical_rank = final_fit.numerical_rank,
      .weighted_objective = final_fit.objective,
      .weighted_rmse = std::sqrt(final_fit.objective),
      .histogram_maximum_absolute_error =
          histogram_maximum_absolute,
      .histogram_maximum_relative_error =
          histogram_maximum_relative,
      .validation_maximum_absolute_error =
          validation_maximum_absolute,
      .validation_maximum_relative_error =
          validation_maximum_relative,
  };
}

LaplaceFitResult build_and_fit_laplace_quadrature(
    const std::vector<double>& orbital_energies,
    const std::vector<std::size_t>& active_occupied_indices,
    const std::vector<std::size_t>& virtual_indices,
    const LaplaceFitOptions& options) {
  const DenominatorHistogram histogram =
      options.scheme == LaplaceQuadratureScheme::gauss_laguerre
          ? build_denominator_interval(orbital_energies,
                                       active_occupied_indices,
                                       virtual_indices, options)
          : build_denominator_histogram(orbital_energies,
                                        active_occupied_indices,
                                        virtual_indices, options);
  if(options.mpi_ranks == 1) {
    return fit_laplace_quadrature(histogram, options);
  }
#ifdef MODERNQC_HAS_MPI
  return root_fit_and_broadcast(histogram, options);
#else
  throw std::runtime_error(
      "multiple Laplace fitting ranks require MPI support");
#endif
}

double evaluate_laplace_reciprocal(
    double denominator, const std::vector<double>& nodes,
    const std::vector<double>& weights) {
  if(!std::isfinite(denominator) || denominator <= 0.0 ||
     nodes.empty() || nodes.size() != weights.size()) {
    throw std::invalid_argument(
        "Laplace reciprocal evaluation inputs are invalid");
  }
  long double value = 0.0L;
  for(std::size_t point = 0; point < nodes.size(); ++point) {
    if(!std::isfinite(nodes[point]) || nodes[point] <= 0.0 ||
       !std::isfinite(weights[point])) {
      throw std::invalid_argument(
          "Laplace reciprocal nodes or weights are invalid");
    }
    value +=
        static_cast<long double>(weights[point]) *
        static_cast<long double>(safe_decay(denominator, nodes[point]));
  }
  const double result = static_cast<double>(value);
  if(!std::isfinite(result)) {
    throw std::runtime_error(
        "Laplace reciprocal approximation is nonfinite");
  }
  return result;
}

}  // namespace modernqc::mp2
