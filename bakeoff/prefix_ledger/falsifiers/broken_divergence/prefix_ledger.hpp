#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>
namespace kv {
using TokenId = std::int32_t;
struct ReuseDecision { std::size_t reusable = 0; bool divergent = false; };
class PrefixLedger {
  public:
    void append(TokenId id);
    void append(std::span<const TokenId> ids);
    [[nodiscard]] ReuseDecision plan_reuse(std::span<const TokenId> candidate) const;
    void truncate_last(std::size_t n);
    void truncate_to(std::size_t n);
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::span<const TokenId> tokens() const noexcept;
    void clear() noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
  private:
    std::vector<TokenId> ids_;
};
} // namespace kv
