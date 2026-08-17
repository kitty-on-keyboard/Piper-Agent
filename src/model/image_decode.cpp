#include "src/model/image_decode.hpp"

#include <algorithm>
#include <cctype>
#include <string_view>
#include <vector>

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#endif

namespace lmp::model {
namespace {

std::string lower_ext(std::string_view path) {
    const std::size_t dot = path.rfind('.');
    if (dot == std::string_view::npos) {
        return {};
    }
    std::string e(path.substr(dot + 1));
    for (char& c : e) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return e;
}

#if defined(__APPLE__)

// Every CF object this file creates goes through one of these. ImageIO's C API is
// CFRetain/CFRelease, and a decoder that leaks a CGImage per call leaks megabytes per
// image -- which on a 48 GB host shared with a 19 GB model is not a slow leak.
template <typename T>
class CFRef {
  public:
    CFRef() = default;
    explicit CFRef(T ref) : ref_(ref) {}
    CFRef(const CFRef&) = delete;
    CFRef& operator=(const CFRef&) = delete;
    ~CFRef() {
        if (ref_ != nullptr) {
            CFRelease(ref_);
        }
    }
    [[nodiscard]] T get() const noexcept { return ref_; }
    explicit operator bool() const noexcept { return ref_ != nullptr; }

  private:
    T ref_ = nullptr;
};

bool decode_source(CGImageSourceRef src, long long max_pixels, ImageRGB& out,
                   std::string& error) {
    if (src == nullptr || CGImageSourceGetCount(src) == 0) {
        error = "not a decodable image (no image data found)";
        return false;
    }

    // THE HEADER IS CHECKED BEFORE THE PIXELS EXIST. CGImageSourceCopyPropertiesAtIndex
    // reads dimensions without decompressing, so a file claiming 60000x60000 is refused
    // here rather than after ImageIO has been asked for 10 GB of bitmap.
    CFRef<CFDictionaryRef> props(CGImageSourceCopyPropertiesAtIndex(src, 0, nullptr));
    long long w = 0;
    long long h = 0;
    if (props) {
        const auto read = [&props](CFStringRef key, long long& v) {
            const void* raw = CFDictionaryGetValue(props.get(), key);
            if (raw != nullptr && CFGetTypeID(raw) == CFNumberGetTypeID()) {
                CFNumberGetValue(static_cast<CFNumberRef>(raw), kCFNumberLongLongType, &v);
            }
        };
        read(kCGImagePropertyPixelWidth, w);
        read(kCGImagePropertyPixelHeight, h);
    }
    if (w > 0 && h > 0 && w * h > max_pixels) {
        error = "image is " + std::to_string(w) + "x" + std::to_string(h) + " (" +
                std::to_string(w * h) + " pixels), over the " +
                std::to_string(max_pixels) + " pixel limit for a decoded image";
        return false;
    }

    CFRef<CGImageRef> img(CGImageSourceCreateImageAtIndex(src, 0, nullptr));
    if (!img) {
        error = "the file was recognised as an image but could not be decoded";
        return false;
    }
    const auto iw = static_cast<int>(CGImageGetWidth(img.get()));
    const auto ih = static_cast<int>(CGImageGetHeight(img.get()));
    if (iw <= 0 || ih <= 0) {
        error = "decoded image has a zero dimension";
        return false;
    }
    // Re-checked against the DECODED dimensions: the header is metadata and metadata can
    // disagree with the pixels.
    if (static_cast<long long>(iw) * static_cast<long long>(ih) > max_pixels) {
        error = "decoded image is larger than its header declared, and over the pixel "
                "limit";
        return false;
    }

    // Draw into a known layout rather than trusting the source's: ImageIO will hand back
    // indexed, CMYK, 16-bit or planar data depending on the file, and every one of those
    // reads as garbage if treated as RGB8. RGBA8 premultiplied is the one CGBitmapContext
    // supports everywhere; the alpha is composited and dropped below.
    const std::size_t stride = static_cast<std::size_t>(iw) * 4U;
    std::vector<std::uint8_t> rgba(stride * static_cast<std::size_t>(ih), 0);
    CFRef<CGColorSpaceRef> space(CGColorSpaceCreateDeviceRGB());
    // Widened before the OR: the two constants are different anonymous enums and C++20
    // deprecates the arithmetic between them, which -Werror makes fatal.
    const std::uint32_t bitmap_info =
        static_cast<std::uint32_t>(kCGImageAlphaPremultipliedLast) |
        static_cast<std::uint32_t>(kCGBitmapByteOrder32Big);
    CFRef<CGContextRef> ctx(CGBitmapContextCreate(
        rgba.data(), static_cast<std::size_t>(iw), static_cast<std::size_t>(ih), 8, stride,
        space.get(), bitmap_info));
    if (!ctx) {
        error = "could not allocate a bitmap context for the decoded image";
        return false;
    }
    // WHITE, not transparent-to-black. A screenshot with a rounded transparent corner
    // composited onto black grows a black border the model will describe.
    CGContextSetRGBFillColor(ctx.get(), 1.0, 1.0, 1.0, 1.0);
    CGContextFillRect(ctx.get(), CGRectMake(0, 0, iw, ih));
    CGContextDrawImage(ctx.get(), CGRectMake(0, 0, iw, ih), img.get());

    out.width = iw;
    out.height = ih;
    out.rgb.resize(static_cast<std::size_t>(iw) * static_cast<std::size_t>(ih) * 3U);
    for (std::size_t p = 0, q = 0; p < rgba.size(); p += 4, q += 3) {
        out.rgb[q] = rgba[p];
        out.rgb[q + 1] = rgba[p + 1];
        out.rgb[q + 2] = rgba[p + 2];
    }
    return true;
}

#endif // __APPLE__

} // namespace

bool looks_like_image_path(std::string_view path) {
    static constexpr std::string_view kExts[] = {"png",  "jpg",  "jpeg", "heic", "heif",
                                                 "gif",  "bmp",  "tif",  "tiff", "webp"};
    const std::string e = lower_ext(path);
    return std::find(std::begin(kExts), std::end(kExts), e) != std::end(kExts);
}

#if defined(__APPLE__)

bool decode_image_file(const std::string& path, long long max_pixels, ImageRGB& out,
                       std::string& error) {
    CFRef<CFStringRef> cf_path(CFStringCreateWithBytes(
        nullptr, reinterpret_cast<const UInt8*>(path.data()),
        static_cast<CFIndex>(path.size()), kCFStringEncodingUTF8, false));
    if (!cf_path) {
        error = "path is not valid UTF-8";
        return false;
    }
    CFRef<CFURLRef> url(CFURLCreateWithFileSystemPath(nullptr, cf_path.get(),
                                                      kCFURLPOSIXPathStyle, false));
    if (!url) {
        error = "could not form a URL for " + path;
        return false;
    }
    CFRef<CGImageSourceRef> src(CGImageSourceCreateWithURL(url.get(), nullptr));
    if (!src) {
        error = "could not open " + path + " as an image";
        return false;
    }
    return decode_source(src.get(), max_pixels, out, error);
}

bool decode_image_bytes(const std::uint8_t* data, std::size_t size, long long max_pixels,
                        ImageRGB& out, std::string& error) {
    if (data == nullptr || size == 0) {
        error = "empty image buffer";
        return false;
    }
    CFRef<CFDataRef> cf(CFDataCreate(nullptr, data, static_cast<CFIndex>(size)));
    if (!cf) {
        error = "could not wrap the image bytes";
        return false;
    }
    CFRef<CGImageSourceRef> src(CGImageSourceCreateWithData(cf.get(), nullptr));
    if (!src) {
        error = "the bytes are not a recognised image format";
        return false;
    }
    return decode_source(src.get(), max_pixels, out, error);
}

#else // !__APPLE__

// Not a stub that returns an empty image: a caller that gets `true` and no pixels would
// send a blank patch grid to the tower and get a confident description of nothing.
bool decode_image_file(const std::string& path, long long, ImageRGB&, std::string& error) {
    (void)path;
    error = "image decoding needs Apple's ImageIO, which this platform does not have "
            "(and without MLX there is no model to show an image to)";
    return false;
}

bool decode_image_bytes(const std::uint8_t*, std::size_t, long long, ImageRGB&,
                        std::string& error) {
    error = "image decoding needs Apple's ImageIO, which this platform does not have";
    return false;
}

#endif // __APPLE__

} // namespace lmp::model
