#!/usr/bin/env bash
# Foreground only. Invoke explicitly on the validation laptop.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$(uname -s)" == Darwin ]] || { echo 'Use the portable build instructions on other systems.' >&2; exit 2; }
command -v brew >/dev/null
command -v cmake >/dev/null
command -v python3 >/dev/null
libint_prefix="$(brew --prefix libint)"
eigen_prefix="$(brew --prefix eigen)"
omp_prefix="$(brew --prefix libomp)"
blas_prefix="$(brew --prefix openblas)"
boost_prefix="$(brew --prefix boost)"
test -s "$libint_prefix/lib/cmake/libint2/libint2-config.cmake"
test -s "$eigen_prefix/share/eigen3/cmake/Eigen3Config.cmake"
test -s "$omp_prefix/lib/libomp.dylib"
basis_file="$(find "$libint_prefix/share/libint" -type f -name ano-rcc.g94 | sort | tail -n 1)"
test -n "$basis_file"
export LIBINT_DATA_PATH="$(dirname "$basis_file")"
ram_mib=$(( $(sysctl -n hw.memsize) / 1048576 ))
(( ram_mib >= 16384 )) || { echo 'This validation plan requires at least 16 GiB RAM; do not risk memory pressure.' >&2; exit 2; }
threads="$(sysctl -n hw.physicalcpu)"
(( threads <= 4 )) || threads=4
memory_mib=$((ram_mib / 4))
(( memory_mib <= 8192 )) || memory_mib=8192
export OMP_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
mkdir -p runs
campaign="$(mktemp -d "$PWD/runs/w2-validation.XXXXXX")"
find CMakeLists.txt LICENSE NOTICES.md README.md RELEASE.md VALIDATION.md CHANGELOG.md .gitignore lccsdt apps src include kernels vendor tests scripts examples notices -type f ! -name '._*' ! -name '.DS_Store' ! -name '*.pyc' -print0 | xargs -0 shasum -a 256 > "$campaign/source.sha256"
while IFS= read -r name; do unset "$name"; done < <(compgen -v CCSD_)
echo "VALIDATION DIRECTORY: $campaign"
echo "Physical threads=$threads; additional workspace=$memory_mib MiB; full PM nM; spin symmetry; resident triples"
options=(-DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DLCCSDT_MPI=OFF -DLCCSDT_TESTS=ON
         "-DCMAKE_PREFIX_PATH=$libint_prefix;$eigen_prefix;$omp_prefix;$blas_prefix;$boost_prefix"
         "-DLibint2_DIR=$libint_prefix/lib/cmake/libint2"
         "-DEigen3_DIR=$eigen_prefix/share/eigen3/cmake" -DBLA_VENDOR=OpenBLAS
         '-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp' -DOpenMP_CXX_LIB_NAMES=omp
         "-DOpenMP_omp_LIBRARY=$omp_prefix/lib/libomp.dylib"
         "-DOpenMP_CXX_INCLUDE_DIR=$omp_prefix/include")
cmake -S . -B build/laptop-debug -DCMAKE_BUILD_TYPE=Debug "${options[@]}" 2>&1 | tee "$campaign/configure-debug.log"
cmake --build build/laptop-debug -j 2 2>&1 | tee "$campaign/build-debug.log"
ctest --test-dir build/laptop-debug --output-on-failure 2>&1 | tee "$campaign/tests-debug.log"
cmake -S . -B build/laptop-release -DCMAKE_BUILD_TYPE=Release "${options[@]}" 2>&1 | tee "$campaign/configure-release.log"
cmake --build build/laptop-release -j 2 2>&1 | tee "$campaign/build-release.log"
ctest --test-dir build/laptop-release --output-on-failure 2>&1 | tee "$campaign/tests-release.log"
./lccsdt run examples/w2-avdz.xyz --basis aug-cc-pVDZ --build build/laptop-release --threads "$threads" --memory-mib "$memory_mib" --storage memory --symmetry spin --output "$campaign/calculation"
python3 scripts/check_w2.py "$campaign/calculation" | tee "$campaign/reference-comparison.log"
mkdir "$campaign/restart-check"
for name in geometry.xyz run.json state.chk state.ccsd.chk state.chk.localized-hamiltonian; do
  cp "$campaign/calculation/$name" "$campaign/restart-check/$name"
done
./lccsdt restart "$campaign/restart-check" --triples-only
python3 scripts/check_restart.py "$campaign/calculation" "$campaign/restart-check" | tee "$campaign/restart-comparison.log"
shasum -a 256 -c "$campaign/source.sha256" > "$campaign/source-verification.log"
printf '%s\n' "$campaign" > runs/last-validated.txt
echo "VALIDATION PASSED: $campaign/calculation/W2_VALIDATED.json"
echo 'No public repository was created. Inspect these results before publishing.'
