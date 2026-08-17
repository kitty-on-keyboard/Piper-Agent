#pragma once
//
// File bytes -> RGB8, via the system's ImageIO.
//
// NO VENDORED DECODER, on purpose. The obvious move was stb_image -- single header,
// public domain, the usual choice -- and it is the wrong one here for three reasons:
// this product is Apple-only by construction (MLX, Metal), CI runs macos-15, and an
// image decoder is the single most attacked parser in any program that has one. ImageIO
// is maintained and hardened by the platform, ships in every build we can target, costs
// zero vendored lines, and reads HEIC -- which matters, because that is what a macOS
// screenshot and every photo off an iPhone actually is.
//
// The one thing it does not give us is a Linux build. That is not a regression: without
// MLX there is no model to show an image to, and the gate's image tests run against
// synthesized pixels through image_preprocess.hpp, which has no platform dependency at
// all.
//
// DECODED IMAGES ARE UNTRUSTED INPUT. The bytes come from the workspace, which means
// they can come from a repository the operator merely cloned. `max_pixels` is a real
// bound and not a formality: it is checked against the header BEFORE the pixels are
// materialised, so a decompression bomb is refused rather than allocated.
#include <cstddef>
#include <cstdint>
#include <string>

#include "src/model/image_preprocess.hpp"

namespace lmp::model {

// Whether `path`'s extension names a format this build will attempt. Cheap, and
// deliberately NOT the authority -- decode_image sniffs the actual bytes. It exists so a
// tool can say "that is not an image" without reading the file.
[[nodiscard]] bool looks_like_image_path(std::string_view path);

// Decodes to 8-bit RGB, dropping any alpha (compositing onto white, which is what a
// screenshot with a transparent corner should look like rather than black).
//
// `max_pixels` bounds the DECODED size and is enforced against the file's header first.
// Returns false with `error` set on anything it cannot read; never throws.
[[nodiscard]] bool decode_image_file(const std::string& path, long long max_pixels,
                                     ImageRGB& out, std::string& error);

// The same, for bytes already in memory -- the path a pasted or dragged image will take
// once the surface can carry one.
[[nodiscard]] bool decode_image_bytes(const std::uint8_t* data, std::size_t size,
                                      long long max_pixels, ImageRGB& out,
                                      std::string& error);

} // namespace lmp::model
