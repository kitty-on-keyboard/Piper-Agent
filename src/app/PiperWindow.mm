#import "src/app/PiperWindow.h"

#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <WebKit/WebKit.h>

#include <cstdlib>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#import "src/app/SidecarBridge.h"
#include "src/app/host_helpers.hpp"

namespace {

using json = nlohmann::json;

std::string json_to_std_string(id obj) {
    if (!obj) return "{}";
    if ([obj isKindOfClass:[NSString class]]) {
        return [(NSString*)obj UTF8String];
    }
    NSError* err = nil;
    NSData* data = [NSJSONSerialization dataWithJSONObject:obj options:0 error:&err];
    if (!data || err) return "{}";
    return std::string(static_cast<const char*>(data.bytes), data.length);
}

NSString* ns_string_from_std(const std::string& s) {
    NSString* out = [[NSString alloc] initWithBytes:s.data()
                                             length:s.size()
                                           encoding:NSUTF8StringEncoding];
    return out ? out : @"";
}

std::filesystem::path resolve_ui_bundle_root() {
    NSString* resourcePath = [[NSBundle mainBundle] resourcePath];
    if (resourcePath) {
        std::filesystem::path bundled = std::filesystem::path([resourcePath UTF8String]) / "ui";
        if (std::filesystem::is_directory(bundled)) return bundled;
    }
    return {};
}

} // namespace

@interface PiperSchemeHandler : NSObject <WKURLSchemeHandler>
- (instancetype)initWithUiRoot:(std::filesystem::path)uiRoot;
@end

@implementation PiperSchemeHandler {
    std::filesystem::path _uiRoot;
}

- (instancetype)initWithUiRoot:(std::filesystem::path)uiRoot {
    self = [super init];
    if (self) {
        _uiRoot = std::move(uiRoot);
    }
    return self;
}

- (void)webView:(WKWebView*)webView startURLSchemeTask:(id<WKURLSchemeTask>)urlSchemeTask {
    (void)webView;
    NSURL* url = urlSchemeTask.request.URL;
    NSString* rawPath = url.path ? url.path : @"/index.html";
    const auto resolved = lmp::app::resolve_ui_asset(_uiRoot, [rawPath UTF8String]);
    if (!resolved.ok) {
        NSURLResponse* resp = [[NSHTTPURLResponse alloc] initWithURL:url
                                                          statusCode:404
                                                         HTTPVersion:@"HTTP/1.1"
                                                        headerFields:@{@"Content-Type": @"text/plain; charset=utf-8"}];
        [urlSchemeTask didReceiveResponse:resp];
        [urlSchemeTask didReceiveData:[@"Not found" dataUsingEncoding:NSUTF8StringEncoding]];
        [urlSchemeTask didFinish];
        return;
    }

    NSString* filePath = ns_string_from_std(resolved.path.string());
    NSData* data = [NSData dataWithContentsOfFile:filePath];
    if (!data) {
        [urlSchemeTask didFailWithError:[NSError errorWithDomain:@"PiperScheme"
                                                            code:500
                                                        userInfo:@{NSLocalizedDescriptionKey: @"Read failed"}]];
        return;
    }

    NSString* mime = ns_string_from_std(resolved.mime_type);
    NSDictionary* headers = @{
        @"Content-Type": mime,
        @"Cache-Control": @"no-store",
    };
    NSURLResponse* resp = [[NSHTTPURLResponse alloc] initWithURL:url
                                                      statusCode:200
                                                     HTTPVersion:@"HTTP/1.1"
                                                    headerFields:headers];
    [urlSchemeTask didReceiveResponse:resp];
    [urlSchemeTask didReceiveData:data];
    [urlSchemeTask didFinish];
}

- (void)webView:(WKWebView*)webView stopURLSchemeTask:(id<WKURLSchemeTask>)urlSchemeTask {
    (void)webView;
    (void)urlSchemeTask;
}

@end

@interface PiperWindowController () <WKScriptMessageHandler, WKNavigationDelegate, WKUIDelegate>
@end

@implementation PiperWindowController {
    WKWebView* _webView;
    SidecarBridge* _sidecar;
    NSString* _devURL;
    BOOL _clientReady;
    NSMutableArray<NSString*>* _pendingToWeb;
    dispatch_queue_t _hostQueue;
}

- (instancetype)initWithDevURL:(nullable NSString*)devURL {
    NSRect frame = NSMakeRect(0, 0, 1340, 860);
    NSWindowStyleMask style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                              NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable |
                              NSWindowStyleMaskFullSizeContentView;
    NSWindow* window = [[NSWindow alloc] initWithContentRect:frame
                                                   styleMask:style
                                                     backing:NSBackingStoreBuffered
                                                       defer:NO];
    window.minSize = NSMakeSize(920, 620);
    window.title = @"Piper";
    window.titleVisibility = NSWindowTitleHidden;
    window.titlebarAppearsTransparent = YES;
    window.movableByWindowBackground = NO;
    [window center];
    [window setFrameAutosaveName:@"PiperMainWindow"];

    self = [super initWithWindow:window];
    if (self) {
        _devURL = [devURL copy];
        _clientReady = NO;
        _pendingToWeb = [NSMutableArray array];
        _hostQueue = dispatch_queue_create("com.piper.host.io", DISPATCH_QUEUE_CONCURRENT);
        _sidecar = [[SidecarBridge alloc] init];

        [self setupWebViewInWindow:window];
        [self startSidecar];
        [self loadUI];
    }
    return self;
}

- (void)setupWebViewInWindow:(NSWindow*)window {
    WKWebViewConfiguration* cfg = [[WKWebViewConfiguration alloc] init];
    cfg.preferences.javaScriptCanOpenWindowsAutomatically = NO;

    PiperSchemeHandler* handler = [[PiperSchemeHandler alloc] initWithUiRoot:resolve_ui_bundle_root()];
    [cfg setURLSchemeHandler:handler forURLScheme:@"piper"];
    [cfg.userContentController addScriptMessageHandler:self name:@"piper"];

    _webView = [[WKWebView alloc] initWithFrame:window.contentView.bounds configuration:cfg];
    _webView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    _webView.navigationDelegate = self;
    _webView.UIDelegate = self;
    [_webView setValue:@NO forKey:@"drawsBackground"];
    if (_devURL != nil || std::getenv("PIPER_INSPECTOR") != nullptr) {
        _webView.inspectable = YES;
    }

    window.contentView = _webView;
}

- (void)startSidecar {
    __weak PiperWindowController* weakSelf = self;
    _sidecar.onMessage = ^(NSString* jsonLine) {
        [weakSelf deliverJsonToWeb:jsonLine];
    };
    _sidecar.onExit = ^(int code, NSArray<NSString*>* stderrTail) {
        json tail = json::array();
        for (NSString* line in stderrTail) {
            tail.push_back([line UTF8String]);
        }
        const json note = {
            {"jsonrpc", "2.0"},
            {"method", "host/sidecar_exit"},
            {"params", {{"code", code}, {"stderr_tail", std::move(tail)}}},
        };
        [weakSelf deliverJsonToWeb:ns_string_from_std(note.dump())];
    };

    NSString* err = nil;
    if (![_sidecar startSidecarWithError:&err]) {
        const json note = {
            {"jsonrpc", "2.0"},
            {"method", "host/sidecar_exit"},
            {"params", {{"code", -1}, {"stderr_tail", {err ? [err UTF8String] : "failed to start"}}}},
        };
        [self deliverJsonToWeb:ns_string_from_std(note.dump())];
    }
}

- (void)loadUI {
    _clientReady = NO;
    if (_devURL && _devURL.length > 0) {
        NSURL* url = [NSURL URLWithString:_devURL];
        if (url) {
            [_webView loadRequest:[NSURLRequest requestWithURL:url]];
            return;
        }
    }
    NSURL* appURL = [NSURL URLWithString:@"piper://app/index.html"];
    [_webView loadRequest:[NSURLRequest requestWithURL:appURL]];
}

- (void)reloadWebView {
    [self loadUI];
}

- (void)shutdownSidecar {
    [_sidecar stopSidecar];
}

- (void)emitShortcut:(NSString*)shortcut {
    const json note = {
        {"jsonrpc", "2.0"},
        {"method", "host/shortcut"},
        {"params", {{"name", shortcut ? [shortcut UTF8String] : ""}}},
    };
    [self deliverJsonToWeb:ns_string_from_std(note.dump())];
}

- (void)deliverJsonToWeb:(NSString*)jsonLine {
    if (!jsonLine) return;
    if (!_clientReady) {
        if (_pendingToWeb.count < 256) {
            [_pendingToWeb addObject:jsonLine];
        }
        return;
    }
    const std::string escaped = lmp::app::escape_js_string_arg([jsonLine UTF8String]);
    NSString* script =
        [NSString stringWithFormat:@"window.__piperReceive && window.__piperReceive(%s);", escaped.c_str()];
    [_webView evaluateJavaScript:script completionHandler:nil];
}

- (void)flushPendingMessages {
    if (!_clientReady || _pendingToWeb.count == 0) return;
    NSArray<NSString*>* copy = [_pendingToWeb copy];
    [_pendingToWeb removeAllObjects];
    for (NSString* msg in copy) {
        [self deliverJsonToWeb:msg];
    }
}

- (void)replyId:(const json&)idVal resultJson:(const std::string&)resultJson {
    json payload;
    try {
        payload = json::parse(resultJson);
    } catch (...) {
        payload = json::object();
    }
    const json resp = {{"jsonrpc", "2.0"}, {"id", idVal}, {"result", std::move(payload)}};
    [self deliverJsonToWeb:ns_string_from_std(resp.dump())];
}

- (void)replyId:(const json&)idVal error:(const std::string&)message {
    const json resp = {
        {"jsonrpc", "2.0"},
        {"id", idVal},
        {"error", {{"code", -32000}, {"message", message}}},
    };
    [self deliverJsonToWeb:ns_string_from_std(resp.dump())];
}

#pragma mark - WKScriptMessageHandler

- (void)userContentController:(WKUserContentController*)userContentController
      didReceiveScriptMessage:(WKScriptMessage*)message {
    (void)userContentController;
    const std::string raw = json_to_std_string(message.body);
    json req;
    try {
        req = json::parse(raw);
    } catch (...) {
        return;
    }
    if (!req.is_object()) return;
    const std::string method = req.value("method", std::string{});
    if (method.empty()) return;

    if (method.rfind("lmp/", 0) == 0) {
        [_sidecar sendRawJsonLine:ns_string_from_std(raw)];
        return;
    }

    if (method.rfind("host/", 0) != 0) {
        return;
    }

    const json reqId = req.contains("id") ? req["id"] : json(nullptr);
    const json params = (req.contains("params") && req["params"].is_object()) ? req["params"] : json::object();

    if (method == "host/window_drag") {
        NSEvent* ev = [NSApp currentEvent];
        if (ev && self.window) {
            [self.window performWindowDragWithEvent:ev];
        }
        if (!reqId.is_null()) [self replyId:reqId resultJson:R"({"ok":true})"];
        return;
    }

    if (method == "host/init") {
        _clientReady = YES;
        const char* qwenEnv = std::getenv("LMP_QWEN_DIR");
        const auto settingsPath = lmp::app::default_settings_path();
        json settingsObj = json::object();
        try {
            settingsObj = json::parse(lmp::app::load_settings_json(settingsPath));
        } catch (...) {
            settingsObj = json::object();
        }
        const char* homeEnv = std::getenv("HOME");
        const json res = {
            {"app_version", "2.0.0"},
            {"sidecar_running", _sidecar.running ? true : false},
            {"qwen_dir", qwenEnv ? std::string(qwenEnv) : std::string{}},
            {"home_dir", homeEnv ? std::string(homeEnv) : std::string{}},
            {"settings_path", settingsPath.string()},
            {"settings", std::move(settingsObj)},
        };
        [self replyId:reqId resultJson:res.dump()];
        [self flushPendingMessages];
        return;
    }

    if (method == "host/save_settings") {
        const json settings = params.contains("settings") ? params["settings"] : json::object();
        std::string err;
        const bool ok = lmp::app::save_settings_json(lmp::app::default_settings_path(), settings.dump(2), err);
        [self replyId:reqId resultJson:json({{"ok", ok}, {"error", err}}).dump()];
        return;
    }

    if (method == "host/pick_folder") {
        NSOpenPanel* panel = [NSOpenPanel openPanel];
        panel.canChooseFiles = NO;
        panel.canChooseDirectories = YES;
        panel.allowsMultipleSelection = NO;
        panel.canCreateDirectories = YES;
        const std::string title = params.value("title", std::string("Choose Folder"));
        panel.title = ns_string_from_std(title);
        const std::string defPath = params.value("default_path", std::string{});
        if (!defPath.empty()) {
            panel.directoryURL = [NSURL fileURLWithPath:ns_string_from_std(defPath)];
        }
        [panel beginSheetModalForWindow:self.window
                      completionHandler:^(NSModalResponse result) {
                          if (result == NSModalResponseOK && panel.URL != nil) {
                              const std::string picked = panel.URL.path ? [panel.URL.path UTF8String] : "";
                              [self replyId:reqId
                                  resultJson:json({{"path", picked}, {"cancelled", false}}).dump()];
                          } else {
                              [self replyId:reqId
                                  resultJson:json({{"path", ""}, {"cancelled", true}}).dump()];
                          }
                      }];
        return;
    }

    if (method == "host/reveal_in_finder") {
        const std::string path = params.value("path", std::string{});
        if (!path.empty()) {
            NSURL* url = [NSURL fileURLWithPath:ns_string_from_std(path)];
            if (url) [[NSWorkspace sharedWorkspace] activateFileViewerSelectingURLs:@[url]];
        }
        [self replyId:reqId resultJson:R"({"ok":true})"];
        return;
    }

    if (method == "host/open_external") {
        const std::string urlStr = params.value("url", std::string{});
        NSURLComponents* comps = [NSURLComponents componentsWithString:ns_string_from_std(urlStr)];
        NSString* scheme = comps.scheme.lowercaseString;
        if (comps && ([scheme isEqualToString:@"http"] || [scheme isEqualToString:@"https"])) {
            [[NSWorkspace sharedWorkspace] openURL:comps.URL];
            [self replyId:reqId resultJson:R"({"ok":true})"];
        } else {
            [self replyId:reqId error:"only http/https external URLs are allowed"];
        }
        return;
    }

    if (method == "host/restart_sidecar") {
        [_sidecar stopSidecar];
        NSString* err = nil;
        const BOOL ok = [_sidecar startSidecarWithError:&err];
        [self replyId:reqId
            resultJson:json({{"ok", ok ? true : false}, {"error", err ? [err UTF8String] : ""}}).dump()];
        return;
    }

    __weak PiperWindowController* weakSelf = self;
    dispatch_async(_hostQueue, ^{
        std::string resultJson = "{}";
        if (method == "host/scan_models") {
            std::vector<std::filesystem::path> roots;
            if (params.contains("roots") && params["roots"].is_array()) {
                for (const auto& r : params["roots"]) {
                    if (r.is_string() && !r.get<std::string>().empty()) {
                        roots.emplace_back(r.get<std::string>());
                    }
                }
            }
            if (const char* qwenEnv = std::getenv("LMP_QWEN_DIR"); qwenEnv && *qwenEnv) {
                roots.emplace_back(qwenEnv);
            }
            if (const char* home = std::getenv("HOME"); home && *home) {
                const std::filesystem::path hp(home);
                roots.push_back(hp / "Models");
                roots.push_back(hp / "models");
                roots.push_back(hp / ".lmstudio" / "models");
            }
            resultJson = lmp::app::scan_models_json(roots);
        } else if (method == "host/workspace_diff") {
            resultJson = lmp::app::workspace_diff_json(params.value("workspace_root", std::string{}));
        } else if (method == "host/workspace_files") {
            resultJson = lmp::app::workspace_files_json(
                params.value("workspace_root", std::string{}), params.value("limit", 2000));
        } else if (method == "host/read_file_preview") {
            resultJson = lmp::app::read_file_preview_json(
                params.value("workspace_root", std::string{}),
                params.value("path", std::string{}),
                static_cast<std::size_t>(std::max(1024, params.value("max_bytes", 262144))));
        } else if (method == "host/save_image_attachment") {
            resultJson = lmp::app::save_image_attachment_json(
                params.value("workspace_root", std::string{}),
                params.value("ext", std::string{}),
                params.value("base64_data", std::string{}));
        } else {
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf replyId:reqId error:("unknown host method: " + method)];
            });
            return;
        }

        dispatch_async(dispatch_get_main_queue(), ^{
            [weakSelf replyId:reqId resultJson:resultJson];
        });
    });
}

#pragma mark - WKNavigationDelegate

- (void)webView:(WKWebView*)webView
    decidePolicyForNavigationAction:(WKNavigationAction*)navigationAction
                    decisionHandler:(void (^)(WKNavigationActionPolicy))decisionHandler {
    (void)webView;
    NSURL* target = navigationAction.request.URL;
    NSString* scheme = target.scheme.lowercaseString;
    if ([scheme isEqualToString:@"piper"] || [scheme isEqualToString:@"about"]) {
        decisionHandler(WKNavigationActionPolicyAllow);
        return;
    }
    if (_devURL && target.absoluteString) {
        NSURL* dev = [NSURL URLWithString:_devURL];
        if (dev && [target.host isEqualToString:dev.host] &&
            [target.port isEqualToNumber:dev.port ? dev.port : @80] &&
            [scheme isEqualToString:@"http"]) {
            decisionHandler(WKNavigationActionPolicyAllow);
            return;
        }
    }
    if ([scheme isEqualToString:@"http"] || [scheme isEqualToString:@"https"]) {
        [[NSWorkspace sharedWorkspace] openURL:target];
    }
    decisionHandler(WKNavigationActionPolicyCancel);
}

@end
