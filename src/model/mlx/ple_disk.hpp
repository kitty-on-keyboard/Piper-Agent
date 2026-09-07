#ifndef LLM_MLX_PLE_DISK_HPP
#define LLM_MLX_PLE_DISK_HPP

#if LMP_HAVE_MLX

// Row gather for Flash-Next's ~32 GB n-gram table without handing the table to MLX.
//
// `mx::load_safetensors` on those shards creates Load arrays whose nbytes still count
// against MLX active memory. Skip-eval was not enough: a real load peaked at 68 GB
// (38.6 GiB eval'd backbone+experts + 29.8 GiB PLE). This reader `pread`s only the
// packed rows a token needs and dequantizes those.

#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unistd.h>
#include <vector>

#include <simdjson.h>

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;

class PleDiskStore {
public:
    bool load(const std::string& model_dir) {
        close_fds();
        shards_.clear();
        dir_ = model_dir;
        const std::string path = model_dir + "/ple-store.json";
        std::ifstream in(path);
        if (!in) {
            return false;
        }
        std::ostringstream buf;
        buf << in.rdbuf();
        simdjson::dom::parser parser;
        simdjson::dom::element root;
        if (parser.parse(buf.str()).get(root)) {
            return false;
        }
        simdjson::dom::object quant;
        bits_ = 4;
        group_size_ = 32;
        mode_ = "affine";
        if (!root["quantization"].get_object().get(quant)) {
            int64_t bits = bits_;
            int64_t gs = group_size_;
            (void)quant["bits"].get_int64().get(bits);
            (void)quant["group_size"].get_int64().get(gs);
            bits_ = static_cast<int>(bits);
            group_size_ = static_cast<int>(gs);
            std::string_view mode;
            if (!quant["mode"].get_string().get(mode)) {
                mode_ = std::string(mode);
            }
        }
        int64_t row_width = 0;
        (void)root["row_width"].get_int64().get(row_width);
        row_width_ = static_cast<int>(row_width);
        simdjson::dom::array shards;
        if (root["shards"].get_array().get(shards)) {
            return false;
        }
        for (simdjson::dom::element e : shards) {
            simdjson::dom::object obj;
            if (e.get_object().get(obj)) {
                continue;
            }
            Shard s;
            int64_t rs = 0;
            int64_t rc = 0;
            (void)obj["row_start"].get_int64().get(rs);
            (void)obj["row_count"].get_int64().get(rc);
            s.row_start = rs;
            s.row_count = rc;
            if (!read_tensor(obj, "weight", s.weight) || !read_tensor(obj, "scales", s.scales) ||
                !read_tensor(obj, "biases", s.biases)) {
                continue;
            }
            shards_.push_back(std::move(s));
        }
        return !shards_.empty();
    }

    [[nodiscard]] bool loaded() const noexcept { return !shards_.empty(); }
    [[nodiscard]] int row_width() const noexcept { return row_width_; }
    [[nodiscard]] int n_shards() const noexcept { return static_cast<int>(shards_.size()); }

    mx::array gather_rows(const std::vector<std::int64_t>& gids, int row_width,
                          int n_shards, std::int64_t rows_per_shard) const {
        const int n = static_cast<int>(gids.size());
        const int width = row_width > 0 ? row_width : row_width_;
        std::vector<float> out(static_cast<std::size_t>(n * width), 0.0f);
        if (n == 0 || shards_.empty() || n_shards <= 0) {
            return mx::array(out.data(), {n, width});
        }
        std::vector<std::vector<Hit>> by_shard(static_cast<std::size_t>(n_shards));
        for (int i = 0; i < n; ++i) {
            const std::int64_t gid = gids[static_cast<std::size_t>(i)];
            const int shard = static_cast<int>(gid / rows_per_shard);
            const int row = static_cast<int>(gid % rows_per_shard);
            if (shard < 0 || shard >= n_shards) {
                continue;
            }
            by_shard[static_cast<std::size_t>(shard)].push_back({i, row});
        }
        for (int shard = 0; shard < n_shards && shard < static_cast<int>(shards_.size());
             ++shard) {
            const auto& hits = by_shard[static_cast<std::size_t>(shard)];
            if (hits.empty()) {
                continue;
            }
            mx::array gathered = gather_shard(shards_[static_cast<std::size_t>(shard)], hits);
            gathered = mx::contiguous(mx::astype(gathered, mx::float32));
            mx::eval(gathered);
            const int dim = static_cast<int>(gathered.shape().back());
            const int copy = std::min(dim, width);
            const float* src = gathered.data<float>();
            for (std::size_t j = 0; j < hits.size(); ++j) {
                std::copy(src + j * static_cast<std::size_t>(dim),
                          src + j * static_cast<std::size_t>(dim) + static_cast<std::size_t>(copy),
                          out.data() + static_cast<std::size_t>(hits[j].out_i * width));
            }
        }
        return mx::array(out.data(), {n, width});
    }

    ~PleDiskStore() { close_fds(); }
    PleDiskStore() = default;
    PleDiskStore(const PleDiskStore&) = delete;
    PleDiskStore& operator=(const PleDiskStore&) = delete;
    PleDiskStore(PleDiskStore&&) = default;
    PleDiskStore& operator=(PleDiskStore&&) = default;

private:
    struct Tensor {
        std::string file;
        std::uint64_t offset{0};
        int cols{0};
        int elem_bytes{0};
    };
    struct Shard {
        std::int64_t row_start{0};
        std::int64_t row_count{0};
        Tensor weight;
        Tensor scales;
        Tensor biases;
    };
    struct Hit {
        int out_i;
        int row;
    };

    static bool read_tensor(simdjson::dom::object& obj, const char* name, Tensor& t) {
        simdjson::dom::object spec;
        if (obj[name].get_object().get(spec)) {
            return false;
        }
        std::string_view file;
        if (spec["file"].get_string().get(file)) {
            return false;
        }
        t.file = std::string(file);
        int64_t off = 0;
        (void)spec["offset"].get_int64().get(off);
        t.offset = static_cast<std::uint64_t>(off);
        simdjson::dom::array shape;
        if (spec["shape"].get_array().get(shape)) {
            return false;
        }
        int i = 0;
        for (simdjson::dom::element d : shape) {
            int64_t v = 0;
            (void)d.get_int64().get(v);
            if (i == 1) {
                t.cols = static_cast<int>(v);
            }
            ++i;
        }
        std::string_view dtype;
        if (!spec["dtype"].get_string().get(dtype)) {
            if (dtype == "U32" || dtype == "I32" || dtype == "F32") {
                t.elem_bytes = 4;
            } else if (dtype == "BF16" || dtype == "F16") {
                t.elem_bytes = 2;
            } else {
                t.elem_bytes = 1;
            }
        } else {
            t.elem_bytes = 4;
        }
        return t.cols > 0 && t.elem_bytes > 0;
    }

    int fd_for(const std::string& file) const {
        const auto it = fds_.find(file);
        if (it != fds_.end()) {
            return it->second;
        }
        const std::string path = dir_ + "/" + file;
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("PLE shard missing: " + path);
        }
        fds_.emplace(file, fd);
        return fd;
    }

    static void pread_fully(int fd, std::uint64_t off, void* dst, std::size_t n) {
        auto* p = static_cast<char*>(dst);
        std::size_t left = n;
        while (left > 0) {
            const ssize_t r = ::pread(fd, p, left, static_cast<off_t>(off));
            if (r <= 0) {
                throw std::runtime_error("PLE pread failed");
            }
            p += r;
            off += static_cast<std::uint64_t>(r);
            left -= static_cast<std::size_t>(r);
        }
    }

    void read_rows(const Tensor& t, const std::vector<Hit>& hits, void* dst) const {
        const int fd = fd_for(t.file);
        const std::size_t row_bytes =
            static_cast<std::size_t>(t.cols) * static_cast<std::size_t>(t.elem_bytes);
        auto* p = static_cast<char*>(dst);
        for (const Hit& h : hits) {
            const std::uint64_t off =
                t.offset + static_cast<std::uint64_t>(h.row) * row_bytes;
            pread_fully(fd, off, p, row_bytes);
            p += row_bytes;
        }
    }

    mx::array gather_shard(const Shard& shard, const std::vector<Hit>& hits) const {
        const int n = static_cast<int>(hits.size());
        std::vector<std::uint32_t> w(
            static_cast<std::size_t>(n) * static_cast<std::size_t>(shard.weight.cols));
        std::vector<std::uint16_t> scales(
            static_cast<std::size_t>(n) * static_cast<std::size_t>(shard.scales.cols));
        std::vector<std::uint16_t> biases(
            static_cast<std::size_t>(n) * static_cast<std::size_t>(shard.biases.cols));
        read_rows(shard.weight, hits, w.data());
        read_rows(shard.scales, hits, scales.data());
        read_rows(shard.biases, hits, biases.data());
        mx::array w_arr(w.data(), {n, shard.weight.cols}, mx::uint32);
        mx::array s_arr(reinterpret_cast<const mx::bfloat16_t*>(scales.data()),
                        {n, shard.scales.cols}, mx::bfloat16);
        mx::array b_arr(reinterpret_cast<const mx::bfloat16_t*>(biases.data()),
                        {n, shard.biases.cols}, mx::bfloat16);
        return mx::dequantize(w_arr, s_arr, b_arr, group_size_, bits_, mode_);
    }

    void close_fds() {
        for (auto& [_, fd] : fds_) {
            if (fd >= 0) {
                ::close(fd);
            }
        }
        fds_.clear();
    }

    std::string dir_;
    std::vector<Shard> shards_;
    mutable std::unordered_map<std::string, int> fds_;
    int bits_{4};
    int group_size_{32};
    int row_width_{160};
    std::string mode_{"affine"};
};

} // namespace lmp::model::mlxl

#endif
#endif
