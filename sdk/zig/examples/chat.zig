const std = @import("std");
const infergo = @import("infergo");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const path = if (args.len > 1) args[1] else "model.gguf";
    var llm = try infergo.LLM.init(path, -1, 4096, 1, 2048);
    defer llm.deinit();

    var tok_buf: [4096]i32 = undefined;
    const n = try llm.tokenize("What is Zig?", true, &tok_buf);

    var out: [8192]u8 = undefined;
    const text = try llm.generate(tok_buf[0..n], 64, 0.7, 0.9, &out);
    std.debug.print("{s}\n", .{text});
}
