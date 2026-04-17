const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const infergo = b.addModule(.{
        .name = "infergo",
        .root_source_file = b.path("src/infergo.zig"),
        .target = target,
        .optimize = optimize,
    });

    infergo.addIncludePath(.{ .cwd_relative = "../../cpp/include" });
    infergo.linkSystemLibrary("infer_api");

    // Example
    const chat = b.addExecutable(.{
        .name = "chat",
        .root_source_file = b.path("examples/chat.zig"),
        .target = target,
        .optimize = optimize,
    });
    chat.root_module.addImport("infergo", infergo);
    chat.linkSystemLibrary("infer_api");
    b.installArtifact(chat);
}
