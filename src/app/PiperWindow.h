#pragma once

#import <Cocoa/Cocoa.h>

NS_ASSUME_NONNULL_BEGIN

@interface PiperWindowController : NSWindowController

- (instancetype)initWithDevURL:(nullable NSString*)devURL;
- (void)emitShortcut:(NSString*)shortcut;
- (void)reloadWebView;
- (void)shutdownSidecar;

@end

NS_ASSUME_NONNULL_END
