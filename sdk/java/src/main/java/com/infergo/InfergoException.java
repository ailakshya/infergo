package com.infergo;

/**
 * Exception thrown by infergo native operations.
 * Wraps error codes and messages from the C API.
 */
public class InfergoException extends RuntimeException {

    private final int errorCode;

    public InfergoException(String message) {
        super(message);
        this.errorCode = -1;
    }

    public InfergoException(int errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public InfergoException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = -1;
    }

    /**
     * Returns the native error code, or -1 if not available.
     */
    public int getErrorCode() {
        return errorCode;
    }
}
