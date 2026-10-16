#pragma once

#include <string>
#include <string_view>

namespace modernqc::basis {

[[nodiscard]] std::string canonical_basis_name(std::string_view name);

}  // namespace modernqc::basis
