#include <string>
#include <vector>

// Joins the parts with `sep` between them.
std::string join(const std::vector<std::string>& parts, const std::string& sep) {
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            out += sep;
        }
        out += parts[i]
    }
    return out;
}

int main() {
    const std::vector<std::string> parts = {"a", "b", "c"};
    return join(parts, ",").size() == 5 ? 0 : 1;
}
