#include "moetrace.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace moetrace {
namespace {

constexpr std::size_t kWindows = 4;
constexpr int kWindowK[kWindows] = {1, 2, 4, 8};
constexpr std::size_t kMaxHistory = 8;

// 256 experts as four words. The window union is four ORs and four popcounts, which is why
// unique-in-window costs nothing worth measuring.
struct Bits {
    std::uint64_t w[4]{};
    void set(int i) { w[i >> 6] |= (1ULL << (i & 63)); }
    void clear() { w[0] = w[1] = w[2] = w[3] = 0; }
    [[nodiscard]] int popcount() const {
        return std::popcount(w[0]) + std::popcount(w[1]) + std::popcount(w[2]) +
               std::popcount(w[3]);
    }
    void or_with(const Bits& o) {
        w[0] |= o.w[0]; w[1] |= o.w[1]; w[2] |= o.w[2]; w[3] |= o.w[3];
    }
    [[nodiscard]] int and_popcount(const Bits& o) const {
        return std::popcount(w[0] & o.w[0]) + std::popcount(w[1] & o.w[1]) +
               std::popcount(w[2] & o.w[2]) + std::popcount(w[3] & o.w[3]);
    }
};

struct LayerAcc {
    std::vector<std::uint64_t> counts;
    Bits history[kMaxHistory];
    std::size_t seen = 0;
    bool have_prev = false;
    Bits prev;
    double overlap_sum = 0.0;
    std::size_t overlap_n = 0;
    double unique_sum[kWindows]{};
    std::size_t unique_n[kWindows]{};
    std::size_t selections = 0;
};

double gini_of(const std::vector<std::uint64_t>& counts) {
    std::vector<std::uint64_t> c = counts;
    std::sort(c.begin(), c.end());
    const auto n = static_cast<double>(c.size());
    long double total = 0.0L;
    long double weighted = 0.0L;
    for (std::size_t i = 0; i < c.size(); ++i) {
        total += static_cast<long double>(c[i]);
        weighted += static_cast<long double>(i + 1) * static_cast<long double>(c[i]);
    }
    if (total <= 0.0L) {
        return 0.0;
    }
    // Standard sorted-sample Gini.
    const long double g = (2.0L * weighted) / (n * total) - (n + 1.0L) / n;
    return static_cast<double>(g < 0.0L ? 0.0L : g);
}

// One JSONL record. Fixed schema, so this is a scanner rather than a JSON parser: find the
// key, take the integer after the colon. Anything unexpected is a malformed line.
bool parse_line(const char* p, const char* end, int& step, int& layer,
                std::vector<int>& experts, int num_experts) {
    const auto find_key = [&](const char* key, std::size_t klen) -> const char* {
        for (const char* q = p; q + klen <= end; ++q) {
            if (std::equal(key, key + klen, q)) {
                const char* r = q + klen;
                while (r < end && (*r == '"' || *r == ':' || *r == ' ')) {
                    ++r;
                }
                return r;
            }
        }
        return nullptr;
    };
    const char* s = find_key("\"step\"", 6);
    const char* l = find_key("\"layer\"", 7);
    const char* e = find_key("\"experts\"", 9);
    if (s == nullptr || l == nullptr || e == nullptr) {
        return false;
    }
    if (std::from_chars(s, end, step).ec != std::errc{}) {
        return false;
    }
    if (std::from_chars(l, end, layer).ec != std::errc{}) {
        return false;
    }
    if (layer < 0) {
        return false;
    }
    if (e >= end || *e != '[') {
        return false;
    }
    experts.clear();
    const char* q = e + 1;
    while (q < end && *q != ']') {
        int v = 0;
        const auto res = std::from_chars(q, end, v);
        if (res.ec != std::errc{}) {
            return false;
        }
        if (v < 0 || v >= num_experts) {
            return false; // out of range is malformed, not clamped
        }
        experts.push_back(v);
        q = res.ptr;
        while (q < end && (*q == ',' || *q == ' ')) {
            ++q;
        }
    }
    if (q >= end || *q != ']') {
        return false;
    }
    if (experts.empty()) {
        return false;
    }
    // Duplicates within a record would corrupt every set statistic downstream.
    for (std::size_t i = 1; i < experts.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (experts[i] == experts[j]) {
                return false;
            }
        }
    }
    return true;
}

class Analyser {
  public:
    explicit Analyser(int num_experts) : num_experts_(num_experts) {}

    void feed(const char* p, const char* end, std::size_t line_no) {
        int step = 0, layer = 0;
        if (!parse_line(p, end, step, layer, scratch_, num_experts_)) {
            ++stats_.malformed_lines;
            (void)line_no;
            return;
        }
        if (static_cast<std::size_t>(layer) >= layers_.size()) {
            layers_.resize(static_cast<std::size_t>(layer) + 1);
        }
        LayerAcc& a = layers_[static_cast<std::size_t>(layer)];
        if (a.counts.empty()) {
            a.counts.assign(static_cast<std::size_t>(num_experts_), 0);
        }
        Bits cur;
        for (int x : scratch_) {
            cur.set(x);
            ++a.counts[static_cast<std::size_t>(x)];
        }
        a.selections += scratch_.size();

        if (a.have_prev) {
            a.overlap_sum += static_cast<double>(cur.and_popcount(a.prev)) /
                             static_cast<double>(scratch_.size());
            ++a.overlap_n;
        }
        a.prev = cur;
        a.have_prev = true;

        a.history[a.seen % kMaxHistory] = cur;
        ++a.seen;
        for (std::size_t wi = 0; wi < kWindows; ++wi) {
            const auto k = static_cast<std::size_t>(kWindowK[wi]);
            if (a.seen < k) {
                continue;
            }
            Bits u;
            for (std::size_t j = 0; j < k; ++j) {
                u.or_with(a.history[(a.seen - 1 - j) % kMaxHistory]);
            }
            a.unique_sum[wi] += static_cast<double>(u.popcount()) /
                                (static_cast<double>(scratch_.size()) * static_cast<double>(k));
            ++a.unique_n[wi];
        }
        max_step_ = std::max(max_step_, static_cast<std::size_t>(step) + 1);
    }

    TraceStats finish() {
        std::vector<std::uint64_t> all(static_cast<std::size_t>(num_experts_), 0);
        double ov = 0.0;
        std::size_t ovn = 0;
        double us[kWindows]{};
        std::size_t usn[kWindows]{};
        std::size_t cold_total = 0, sel_total = 0;

        for (std::size_t li = 0; li < layers_.size(); ++li) {
            LayerAcc& a = layers_[li];
            if (a.counts.empty()) {
                continue;
            }
            LayerStats s;
            s.layer = static_cast<int>(li);
            s.gini = gini_of(a.counts);
            s.selections = a.selections;
            s.mean_consecutive_overlap = a.overlap_n ? a.overlap_sum / double(a.overlap_n) : 0.0;
            for (std::size_t wi = 0; wi < kWindows; ++wi) {
                s.unique_ratio[wi] = a.unique_n[wi] ? a.unique_sum[wi] / double(a.unique_n[wi]) : 0.0;
                s.breakeven_accepted[wi] = s.unique_ratio[wi] * double(kWindowK[wi]) - 1.0;
            }
            s.cold_experts = static_cast<std::size_t>(
                std::count(a.counts.begin(), a.counts.end(), std::uint64_t{0}));
            stats_.per_layer.push_back(s);

            for (std::size_t i = 0; i < all.size(); ++i) {
                all[i] += a.counts[i];
            }
            ov += a.overlap_sum;
            ovn += a.overlap_n;
            for (std::size_t wi = 0; wi < kWindows; ++wi) {
                us[wi] += a.unique_sum[wi];
                usn[wi] += a.unique_n[wi];
            }
            cold_total += s.cold_experts;
            sel_total += s.selections;
        }

        LayerStats& g = stats_.aggregate;
        g.layer = -1;
        g.gini = gini_of(all);
        g.mean_consecutive_overlap = ovn ? ov / double(ovn) : 0.0;
        for (std::size_t wi = 0; wi < kWindows; ++wi) {
            g.unique_ratio[wi] = usn[wi] ? us[wi] / double(usn[wi]) : 0.0;
            g.breakeven_accepted[wi] = g.unique_ratio[wi] * double(kWindowK[wi]) - 1.0;
        }
        g.cold_experts = cold_total;
        g.selections = sel_total;
        stats_.steps = max_step_;
        return std::move(stats_);
    }

  private:
    int num_experts_;
    std::vector<LayerAcc> layers_;
    std::vector<int> scratch_;
    TraceStats stats_;
    std::size_t max_step_ = 0;
};

} // namespace

TraceStats analyse_lines(std::span<const std::string> lines, int num_experts) {
    Analyser a(num_experts);
    std::size_t n = 0;
    for (const std::string& s : lines) {
        a.feed(s.data(), s.data() + s.size(), ++n);
    }
    return a.finish();
}

TraceStats analyse_file(const std::string& path, int num_experts) {
    Analyser a(num_experts);
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return a.finish();
    }
    // Block reads with an in-place scan: no std::string per line, which is the difference
    // against getline at a million lines. `carry` holds a record split across two blocks.
    constexpr std::size_t kBlock = 256U * 1024U;
    std::vector<char> buf(kBlock);
    std::string carry;
    std::size_t line_no = 0;
    while (true) {
        const std::size_t got = std::fread(buf.data(), 1, kBlock, f);
        if (got == 0) {
            break;
        }
        const char* p = buf.data();
        const char* end = p + got;
        while (p < end) {
            const char* nl = static_cast<const char*>(std::memchr(p, '\n', std::size_t(end - p)));
            if (nl == nullptr) {
                carry.append(p, std::size_t(end - p));
                break;
            }
            if (carry.empty()) {
                a.feed(p, nl, ++line_no);
            } else {
                carry.append(p, std::size_t(nl - p));
                a.feed(carry.data(), carry.data() + carry.size(), ++line_no);
                carry.clear();
            }
            p = nl + 1;
        }
    }
    if (!carry.empty()) {
        a.feed(carry.data(), carry.data() + carry.size(), ++line_no);
    }
    std::fclose(f);
    return a.finish();
}

} // namespace moetrace
