#pragma once
//
// Line-oriented unified diff.
//
// Written rather than shelled out to `diff` because this runs inside a tool call on
// content that only exists in the database -- writing two temporary files to ask another
// process what changed would be slower, would leak artifact content onto disk outside
// the store, and would fail differently on a machine with a different diff.
//
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace lmp::pcc {

// Splits on '\n'. A trailing newline does NOT produce a final empty line, so a file and
// the same file without its last newline differ by one line rather than by two.
[[nodiscard]] std::vector<std::string> split_lines(std::string_view text);

// A unified diff with `context` lines of context. Identical inputs produce an empty
// string, which is what lets a caller test "did this change" without a second compare.
//
// Above kDiffCellBudget aligned cells the quadratic alignment is abandoned and the
// result degrades to a single whole-file hunk. That is a real limit, and it is reported
// in the diff's own header line rather than hidden, because a caller who sees a
// suspiciously coarse diff needs to know it hit the bound.
[[nodiscard]] std::string unified_diff(std::string_view a, std::string_view b,
                                       std::string from_label, std::string to_label,
                                       int context = 3);

// 16 million cells: about 64 MiB of alignment table, and roughly a 4000-line file
// against another 4000-line file. Artifacts larger than that are rare and the coarse
// fallback is honest, where a diff that took a minute would just look broken.
inline constexpr std::size_t kDiffCellBudget = 16U * 1000U * 1000U;

} // namespace lmp::pcc
