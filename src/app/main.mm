#import <Cocoa/Cocoa.h>

#include <signal.h>
#include <cstdlib>
#include <string>

#import "src/app/PiperWindow.h"
#include "src/app/host_helpers.hpp"

@interface PiperAppDelegate : NSObject <NSApplicationDelegate>
- (instancetype)initWithDevURL:(nullable NSString*)devURL;
@end

@implementation PiperAppDelegate {
    NSString* _devURL;
    PiperWindowController* _windowController;
}

- (instancetype)initWithDevURL:(nullable NSString*)devURL {
    self = [super init];
    if (self) {
        _devURL = [devURL copy];
    }
    return self;
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
    (void)notification;
    [self buildMainMenu];
    _windowController = [[PiperWindowController alloc] initWithDevURL:_devURL];
    [_windowController showWindow:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender {
    (void)sender;
    return YES;
}

- (void)applicationWillTerminate:(NSNotification*)notification {
    (void)notification;
    [_windowController shutdownSidecar];
}

- (void)buildMainMenu {
    NSMenu* mainMenu = [[NSMenu alloc] initWithTitle:@""];

    NSMenuItem* appItem = [[NSMenuItem alloc] initWithTitle:@"Piper" action:nil keyEquivalent:@""];
    NSMenu* appMenu = [[NSMenu alloc] initWithTitle:@"Piper"];
    [appMenu addItemWithTitle:@"About Piper" action:@selector(orderFrontStandardAboutPanel:) keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Settings…" action:@selector(openSettings:) keyEquivalent:@","];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Hide Piper" action:@selector(hide:) keyEquivalent:@"h"];
    [appMenu addItemWithTitle:@"Quit Piper" action:@selector(terminate:) keyEquivalent:@"q"];
    [appItem setSubmenu:appMenu];
    [mainMenu addItem:appItem];

    NSMenuItem* fileItem = [[NSMenuItem alloc] initWithTitle:@"File" action:nil keyEquivalent:@""];
    NSMenu* fileMenu = [[NSMenu alloc] initWithTitle:@"File"];
    [fileMenu addItemWithTitle:@"New Session" action:@selector(newSession:) keyEquivalent:@"n"];
    [fileMenu addItemWithTitle:@"Open Workspace…" action:@selector(openWorkspace:) keyEquivalent:@"o"];
    [fileItem setSubmenu:fileMenu];
    [mainMenu addItem:fileItem];

    NSMenuItem* editItem = [[NSMenuItem alloc] initWithTitle:@"Edit" action:nil keyEquivalent:@""];
    NSMenu* editMenu = [[NSMenu alloc] initWithTitle:@"Edit"];
    [editMenu addItemWithTitle:@"Undo" action:@selector(undo:) keyEquivalent:@"z"];
    [editMenu addItemWithTitle:@"Redo" action:@selector(redo:) keyEquivalent:@"Z"];
    [editMenu addItem:[NSMenuItem separatorItem]];
    [editMenu addItemWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"];
    [editMenu addItemWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
    [editMenu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
    [editMenu addItemWithTitle:@"Select All" action:@selector(selectAll:) keyEquivalent:@"a"];
    [editItem setSubmenu:editMenu];
    [mainMenu addItem:editItem];

    NSMenuItem* viewItem = [[NSMenuItem alloc] initWithTitle:@"View" action:nil keyEquivalent:@""];
    NSMenu* viewMenu = [[NSMenu alloc] initWithTitle:@"View"];
    [viewMenu addItemWithTitle:@"Toggle Sidebar" action:@selector(toggleSidebar:) keyEquivalent:@"b"];
    [viewMenu addItemWithTitle:@"Toggle Inspector" action:@selector(toggleInspector:) keyEquivalent:@"j"];
    [viewMenu addItem:[NSMenuItem separatorItem]];
    [viewMenu addItemWithTitle:@"Reload UI" action:@selector(reloadView:) keyEquivalent:@"r"];
    [viewItem setSubmenu:viewMenu];
    [mainMenu addItem:viewItem];

    [NSApp setMainMenu:mainMenu];
}

- (void)openSettings:(id)sender {
    (void)sender;
    [_windowController emitShortcut:@"settings"];
}

- (void)newSession:(id)sender {
    (void)sender;
    [_windowController emitShortcut:@"new_session"];
}

- (void)openWorkspace:(id)sender {
    (void)sender;
    [_windowController emitShortcut:@"open_workspace"];
}

- (void)toggleSidebar:(id)sender {
    (void)sender;
    [_windowController emitShortcut:@"toggle_sidebar"];
}

- (void)toggleInspector:(id)sender {
    (void)sender;
    [_windowController emitShortcut:@"toggle_inspector"];
}

- (void)reloadView:(id)sender {
    (void)sender;
    [_windowController reloadWebView];
}

@end

int main(int argc, const char* argv[]) {
    ::signal(SIGPIPE, SIG_IGN);

    std::string devURL;
    if (const char* envUrl = std::getenv("PIPER_DEV_URL"); envUrl && *envUrl) {
        devURL = envUrl;
    }
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--dev-url") {
            devURL = argv[i + 1];
            ++i;
        }
    }
    if (!devURL.empty() && !lmp::app::is_allowed_dev_url(devURL)) {
        NSLog(@"Refusing disallowed dev URL: %s (only http://localhost:* or http://127.0.0.1:* allowed)",
              devURL.c_str());
        return 1;
    }

    @autoreleasepool {
        NSApplication* app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        NSString* nsDevURL =
            devURL.empty() ? nil : [NSString stringWithUTF8String:devURL.c_str()];
        PiperAppDelegate* delegate = [[PiperAppDelegate alloc] initWithDevURL:nsDevURL];
        [app setDelegate:delegate];
        [app run];
    }
    return 0;
}
