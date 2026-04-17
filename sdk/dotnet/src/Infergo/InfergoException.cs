namespace Infergo;

/// <summary>
/// Exception thrown when an infergo native call fails.
/// </summary>
public class InfergoException : Exception
{
    /// <summary>
    /// The native error code returned by the C API, or -1 if unavailable.
    /// </summary>
    public int ErrorCode { get; }

    public InfergoException(string message) : base(message)
    {
        ErrorCode = -1;
    }

    public InfergoException(string message, int errorCode)
        : base(message)
    {
        ErrorCode = errorCode;
    }

    public InfergoException(string message, Exception inner)
        : base(message, inner)
    {
        ErrorCode = -1;
    }

    /// <summary>
    /// Build an exception from the native thread-local error string.
    /// </summary>
    internal static InfergoException FromNative(string context, int errorCode = -1)
    {
        string? nativeMsg = Native.GetLastError();
        string msg = string.IsNullOrEmpty(nativeMsg)
            ? context
            : $"{context}: {nativeMsg}";
        return new InfergoException(msg, errorCode);
    }
}
