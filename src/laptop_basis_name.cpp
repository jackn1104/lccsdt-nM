#include "modernqc/basis/basis_name.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
namespace modernqc::basis {
std::string canonical_basis_name(std::string_view name) {
  std::string x{name};
  std::transform(x.begin(),x.end(),x.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
  if(x=="sto-3g") return "sto-3g";
  if(x=="ccdz"||x=="cc-pvdz") return "cc-pVDZ";
  if(x=="avdz"||x=="aug-cc-pvdz") return "aug-cc-pVDZ";
  if(x=="avtz"||x=="aug-cc-pvtz") return "aug-cc-pVTZ";
  if(x=="avqz"||x=="aug-cc-pvqz") return "aug-cc-pVQZ";
  throw std::invalid_argument(
      "molecular driver supports sto-3g, ccdz, avdz, avtz, and avqz");
}
}
