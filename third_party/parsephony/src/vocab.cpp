#include "parsephony/mask.hpp"

#include <cstdio>
#include <cstring>
#include <memory>

namespace parsephony {

// Format written by scripts/export_vocab.py:
//   u32 count, then per token: u32 id, u8 flags, u16 len, bytes (LE).
bool Vocab::load(const char* path) {
    tokens.clear();
    special.clear();

    std::unique_ptr<FILE, int (*)(FILE*)> f(std::fopen(path, "rb"), &std::fclose);
    if (!f) return false;

    auto read_u32 = [&](uint32_t& v) {
        return std::fread(&v, 4, 1, f.get()) == 1;
    };

    uint32_t count = 0;
    if (!read_u32(count) || count == 0 || count > 4'000'000) return false;

    tokens.resize(count);
    special.assign(count, 0);

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t id;
        uint8_t flags;
        uint16_t len;
        if (!read_u32(id) ||
            std::fread(&flags, 1, 1, f.get()) != 1 ||
            std::fread(&len, 2, 1, f.get()) != 1 ||
            id >= count) {
            tokens.clear(); special.clear();
            return false;
        }
        std::string t(len, '\0');
        if (len && std::fread(t.data(), 1, len, f.get()) != len) {
            tokens.clear(); special.clear();
            return false;
        }
        tokens[id] = std::move(t);
        special[id] = flags & 1;
    }
    return true;
}

} // namespace parsephony
