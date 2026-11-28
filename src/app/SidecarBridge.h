#pragma once

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface SidecarBridge : NSObject

@property(nonatomic, copy, nullable) void (^onMessage)(NSString* jsonLine);
@property(nonatomic, copy, nullable) void (^onExit)(int exitCode, NSArray<NSString*>* stderrTail);
@property(nonatomic, readonly) BOOL running;

- (BOOL)startSidecarWithError:(NSString* _Nullable* _Nullable)errorOut;
- (BOOL)sendRawJsonLine:(NSString*)jsonLine;
- (void)stopSidecar;

@end

NS_ASSUME_NONNULL_END
