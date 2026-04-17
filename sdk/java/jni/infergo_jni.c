/*
 * infergo_jni.c — JNI bridge from Java to the infergo C API.
 *
 * Build: compiled as a shared library (libinfergo_jni.so / .dylib / .dll)
 *        linked against the infergo C library (libinfergo.so).
 *
 * Every JNI function maps directly to an InfergoNative Java method.
 * Opaque C handles (void*) travel as jlong.
 */

#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include "infer_api.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Helper: throw InfergoException with the last C error string.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void throw_infergo(JNIEnv *env, const char *prefix) {
    const char *cerr = infer_last_error_string();
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s: %s", prefix, cerr ? cerr : "unknown error");
    jclass cls = (*env)->FindClass(env, "com/infergo/InfergoException");
    if (cls) {
        (*env)->ThrowNew(env, cls, buf);
    }
}

/* Convert a jlong to a void* pointer. */
#define PTR(h) ((void*)(intptr_t)(h))
/* Convert a void* pointer to a jlong. */
#define HANDLE(p) ((jlong)(intptr_t)(p))

/* ═══════════════════════════════════════════════════════════════════════════
 * Error
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jstring JNICALL
Java_com_infergo_InfergoNative_lastErrorString(JNIEnv *env, jclass cls) {
    (void)cls;
    const char *err = infer_last_error_string();
    if (!err || err[0] == '\0') return NULL;
    return (*env)->NewStringUTF(env, err);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * LLM
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jlong JNICALL
Java_com_infergo_InfergoNative_llmCreate(JNIEnv *env, jclass cls,
                                         jstring jpath, jint nGpuLayers,
                                         jint ctxSize, jint nSeqMax, jint nBatch) {
    (void)cls;
    const char *path = (*env)->GetStringUTFChars(env, jpath, NULL);
    if (!path) return 0;

    InferLLM llm = infer_llm_create(path, (int)nGpuLayers, (int)ctxSize,
                                    (int)nSeqMax, (int)nBatch);
    (*env)->ReleaseStringUTFChars(env, jpath, path);
    return HANDLE(llm);
}

JNIEXPORT void JNICALL
Java_com_infergo_InfergoNative_llmDestroy(JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    infer_llm_destroy(PTR(handle));
}

JNIEXPORT jint JNICALL
Java_com_infergo_InfergoNative_llmVocabSize(JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    return (jint)infer_llm_vocab_size(PTR(handle));
}

JNIEXPORT jint JNICALL
Java_com_infergo_InfergoNative_llmBos(JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    return (jint)infer_llm_bos(PTR(handle));
}

JNIEXPORT jint JNICALL
Java_com_infergo_InfergoNative_llmEos(JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    return (jint)infer_llm_eos(PTR(handle));
}

JNIEXPORT jintArray JNICALL
Java_com_infergo_InfergoNative_llmTokenize(JNIEnv *env, jclass cls,
                                            jlong handle, jstring jtext,
                                            jboolean addBos, jint maxTokens) {
    (void)cls;
    const char *text = (*env)->GetStringUTFChars(env, jtext, NULL);
    if (!text) return NULL;

    int *ids = (int *)malloc(sizeof(int) * (size_t)maxTokens);
    if (!ids) {
        (*env)->ReleaseStringUTFChars(env, jtext, text);
        return NULL;
    }

    int n = infer_llm_tokenize(PTR(handle), text, addBos ? 1 : 0, ids, (int)maxTokens);
    (*env)->ReleaseStringUTFChars(env, jtext, text);

    if (n < 0) {
        free(ids);
        return NULL;
    }

    jintArray result = (*env)->NewIntArray(env, n);
    if (result) {
        (*env)->SetIntArrayRegion(env, result, 0, n, (jint *)ids);
    }
    free(ids);
    return result;
}

JNIEXPORT jstring JNICALL
Java_com_infergo_InfergoNative_llmGenerate(JNIEnv *env, jclass cls,
                                            jlong handle, jintArray jpromptTokens,
                                            jint maxTokens, jfloat temperature,
                                            jfloat topP, jstring jgrammar,
                                            jint maxTextLen) {
    (void)cls;
    jsize nPrompt = (*env)->GetArrayLength(env, jpromptTokens);
    jint *tokens = (*env)->GetIntArrayElements(env, jpromptTokens, NULL);
    if (!tokens) return NULL;

    const char *grammar = NULL;
    if (jgrammar) {
        grammar = (*env)->GetStringUTFChars(env, jgrammar, NULL);
    }

    char *outText = (char *)calloc(1, (size_t)maxTextLen);
    if (!outText) {
        (*env)->ReleaseIntArrayElements(env, jpromptTokens, tokens, JNI_ABORT);
        if (grammar) (*env)->ReleaseStringUTFChars(env, jgrammar, grammar);
        return NULL;
    }

    int outGenTokens = 0;
    int rc = infer_llm_generate(PTR(handle),
                                (const int *)tokens, (int)nPrompt,
                                (int)maxTokens, (float)temperature, (float)topP,
                                grammar,
                                NULL, NULL, /* no callback */
                                outText, (int)maxTextLen,
                                &outGenTokens);

    (*env)->ReleaseIntArrayElements(env, jpromptTokens, tokens, JNI_ABORT);
    if (grammar) (*env)->ReleaseStringUTFChars(env, jgrammar, grammar);

    jstring result = NULL;
    if (rc == 0) {
        result = (*env)->NewStringUTF(env, outText);
    }
    free(outText);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Session (ONNX)
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jlong JNICALL
Java_com_infergo_InfergoNative_sessionCreate(JNIEnv *env, jclass cls,
                                              jstring jprovider, jint deviceId) {
    (void)cls;
    const char *provider = (*env)->GetStringUTFChars(env, jprovider, NULL);
    if (!provider) return 0;

    InferSession s = infer_session_create(provider, (int)deviceId);
    (*env)->ReleaseStringUTFChars(env, jprovider, provider);
    return HANDLE(s);
}

JNIEXPORT jint JNICALL
Java_com_infergo_InfergoNative_sessionLoad(JNIEnv *env, jclass cls,
                                            jlong handle, jstring jpath) {
    (void)cls;
    const char *path = (*env)->GetStringUTFChars(env, jpath, NULL);
    if (!path) return -1;

    int rc = infer_session_load(PTR(handle), path);
    (*env)->ReleaseStringUTFChars(env, jpath, path);
    return (jint)rc;
}

JNIEXPORT void JNICALL
Java_com_infergo_InfergoNative_sessionDestroy(JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    infer_session_destroy(PTR(handle));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Tokenizer
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jlong JNICALL
Java_com_infergo_InfergoNative_tokenizerLoad(JNIEnv *env, jclass cls, jstring jpath) {
    (void)cls;
    const char *path = (*env)->GetStringUTFChars(env, jpath, NULL);
    if (!path) return 0;

    InferTokenizer tok = infer_tokenizer_load(path);
    (*env)->ReleaseStringUTFChars(env, jpath, path);
    return HANDLE(tok);
}

JNIEXPORT void JNICALL
Java_com_infergo_InfergoNative_tokenizerDestroy(JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    infer_tokenizer_destroy(PTR(handle));
}

JNIEXPORT jint JNICALL
Java_com_infergo_InfergoNative_tokenizerVocabSize(JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    return (jint)infer_tokenizer_vocab_size(PTR(handle));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Embedding
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jfloatArray JNICALL
Java_com_infergo_InfergoNative_embedPipeline(JNIEnv *env, jclass cls,
                                              jlong session, jlong tokenizer,
                                              jstring jtext, jint maxDim) {
    (void)cls;
    const char *text = (*env)->GetStringUTFChars(env, jtext, NULL);
    if (!text) return NULL;

    float *vec = (float *)malloc(sizeof(float) * (size_t)maxDim);
    if (!vec) {
        (*env)->ReleaseStringUTFChars(env, jtext, text);
        return NULL;
    }

    int dim = infer_embed_pipeline(PTR(session), PTR(tokenizer), text, vec, (int)maxDim);
    (*env)->ReleaseStringUTFChars(env, jtext, text);

    if (dim < 0) {
        free(vec);
        return NULL;
    }

    jfloatArray result = (*env)->NewFloatArray(env, dim);
    if (result) {
        (*env)->SetFloatArrayRegion(env, result, 0, dim, vec);
    }
    free(vec);
    return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_infergo_InfergoNative_embedBatchPipeline(JNIEnv *env, jclass cls,
                                                   jlong session, jlong tokenizer,
                                                   jobjectArray jtexts, jint maxDim) {
    (void)cls;
    jsize nTexts = (*env)->GetArrayLength(env, jtexts);
    if (nTexts == 0) return NULL;

    /* Build C string array */
    const char **texts = (const char **)malloc(sizeof(char *) * (size_t)nTexts);
    if (!texts) return NULL;

    for (jsize i = 0; i < nTexts; i++) {
        jstring js = (jstring)(*env)->GetObjectArrayElement(env, jtexts, i);
        texts[i] = (*env)->GetStringUTFChars(env, js, NULL);
    }

    size_t totalFloats = (size_t)nTexts * (size_t)maxDim;
    float *vecs = (float *)calloc(totalFloats, sizeof(float));
    if (!vecs) {
        for (jsize i = 0; i < nTexts; i++) {
            jstring js = (jstring)(*env)->GetObjectArrayElement(env, jtexts, i);
            (*env)->ReleaseStringUTFChars(env, js, texts[i]);
        }
        free(texts);
        return NULL;
    }

    int dim = infer_embed_batch_pipeline(PTR(session), PTR(tokenizer),
                                         texts, (int)nTexts, vecs, (int)maxDim);

    /* Release strings */
    for (jsize i = 0; i < nTexts; i++) {
        jstring js = (jstring)(*env)->GetObjectArrayElement(env, jtexts, i);
        (*env)->ReleaseStringUTFChars(env, js, texts[i]);
    }
    free(texts);

    if (dim < 0) {
        free(vecs);
        return NULL;
    }

    jsize flatLen = nTexts * dim;
    jfloatArray result = (*env)->NewFloatArray(env, flatLen);
    if (result) {
        (*env)->SetFloatArrayRegion(env, result, 0, flatLen, vecs);
    }
    free(vecs);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * VectorDB
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jlong JNICALL
Java_com_infergo_InfergoNative_vectordbCreate(JNIEnv *env, jclass cls,
                                               jint dim, jint M, jint efConstruction) {
    (void)env; (void)cls;
    InferVectorDB db = infer_vectordb_create((int)dim, (int)M, (int)efConstruction);
    return HANDLE(db);
}

JNIEXPORT jint JNICALL
Java_com_infergo_InfergoNative_vectordbInsert(JNIEnv *env, jclass cls,
                                               jlong handle, jlong id,
                                               jfloatArray jvec, jstring jmeta) {
    (void)cls;
    jfloat *vec = (*env)->GetFloatArrayElements(env, jvec, NULL);
    if (!vec) return -1;

    const char *meta = NULL;
    if (jmeta) {
        meta = (*env)->GetStringUTFChars(env, jmeta, NULL);
    }

    int rc = infer_vectordb_insert(PTR(handle), (int64_t)id, (const float *)vec, meta);

    (*env)->ReleaseFloatArrayElements(env, jvec, vec, JNI_ABORT);
    if (meta) (*env)->ReleaseStringUTFChars(env, jmeta, meta);
    return (jint)rc;
}

JNIEXPORT jint JNICALL
Java_com_infergo_InfergoNative_vectordbDelete(JNIEnv *env, jclass cls,
                                               jlong handle, jlong id) {
    (void)env; (void)cls;
    return (jint)infer_vectordb_delete(PTR(handle), (int64_t)id);
}

JNIEXPORT jlongArray JNICALL
Java_com_infergo_InfergoNative_vectordbSearchIds(JNIEnv *env, jclass cls,
                                                  jlong handle, jfloatArray jquery,
                                                  jint k, jint efSearch,
                                                  jstring jfilter, jint maxResults) {
    (void)cls;
    jfloat *query = (*env)->GetFloatArrayElements(env, jquery, NULL);
    if (!query) return NULL;

    const char *filter = NULL;
    if (jfilter) {
        filter = (*env)->GetStringUTFChars(env, jfilter, NULL);
    }

    int64_t *ids = (int64_t *)malloc(sizeof(int64_t) * (size_t)maxResults);
    float *dists = (float *)malloc(sizeof(float) * (size_t)maxResults);
    if (!ids || !dists) {
        free(ids); free(dists);
        (*env)->ReleaseFloatArrayElements(env, jquery, query, JNI_ABORT);
        if (filter) (*env)->ReleaseStringUTFChars(env, jfilter, filter);
        return NULL;
    }

    int n = infer_vectordb_search(PTR(handle), (const float *)query,
                                  (int)k, (int)efSearch, filter,
                                  ids, dists, (int)maxResults);

    (*env)->ReleaseFloatArrayElements(env, jquery, query, JNI_ABORT);
    if (filter) (*env)->ReleaseStringUTFChars(env, jfilter, filter);

    if (n < 0) {
        free(ids); free(dists);
        return NULL;
    }

    jlongArray result = (*env)->NewLongArray(env, n);
    if (result) {
        (*env)->SetLongArrayRegion(env, result, 0, n, (jlong *)ids);
    }
    free(ids); free(dists);
    return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_infergo_InfergoNative_vectordbSearchDistances(JNIEnv *env, jclass cls,
                                                        jlong handle, jfloatArray jquery,
                                                        jint k, jint efSearch,
                                                        jstring jfilter, jint maxResults) {
    (void)cls;
    jfloat *query = (*env)->GetFloatArrayElements(env, jquery, NULL);
    if (!query) return NULL;

    const char *filter = NULL;
    if (jfilter) {
        filter = (*env)->GetStringUTFChars(env, jfilter, NULL);
    }

    int64_t *ids = (int64_t *)malloc(sizeof(int64_t) * (size_t)maxResults);
    float *dists = (float *)malloc(sizeof(float) * (size_t)maxResults);
    if (!ids || !dists) {
        free(ids); free(dists);
        (*env)->ReleaseFloatArrayElements(env, jquery, query, JNI_ABORT);
        if (filter) (*env)->ReleaseStringUTFChars(env, jfilter, filter);
        return NULL;
    }

    int n = infer_vectordb_search(PTR(handle), (const float *)query,
                                  (int)k, (int)efSearch, filter,
                                  ids, dists, (int)maxResults);

    (*env)->ReleaseFloatArrayElements(env, jquery, query, JNI_ABORT);
    if (filter) (*env)->ReleaseStringUTFChars(env, jfilter, filter);

    if (n < 0) {
        free(ids); free(dists);
        return NULL;
    }

    jfloatArray result = (*env)->NewFloatArray(env, n);
    if (result) {
        (*env)->SetFloatArrayRegion(env, result, 0, n, dists);
    }
    free(ids); free(dists);
    return result;
}

JNIEXPORT jint JNICALL
Java_com_infergo_InfergoNative_vectordbSave(JNIEnv *env, jclass cls,
                                             jlong handle, jstring jpath) {
    (void)cls;
    const char *path = (*env)->GetStringUTFChars(env, jpath, NULL);
    if (!path) return -1;
    int rc = infer_vectordb_save(PTR(handle), path);
    (*env)->ReleaseStringUTFChars(env, jpath, path);
    return (jint)rc;
}

JNIEXPORT jint JNICALL
Java_com_infergo_InfergoNative_vectordbLoad(JNIEnv *env, jclass cls,
                                             jlong handle, jstring jpath) {
    (void)cls;
    const char *path = (*env)->GetStringUTFChars(env, jpath, NULL);
    if (!path) return -1;
    int rc = infer_vectordb_load(PTR(handle), path);
    (*env)->ReleaseStringUTFChars(env, jpath, path);
    return (jint)rc;
}

JNIEXPORT jint JNICALL
Java_com_infergo_InfergoNative_vectordbSize(JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    return (jint)infer_vectordb_size(PTR(handle));
}

JNIEXPORT void JNICALL
Java_com_infergo_InfergoNative_vectordbFree(JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    infer_vectordb_free(PTR(handle));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BM25
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jlong JNICALL
Java_com_infergo_InfergoNative_bm25Create(JNIEnv *env, jclass cls,
                                           jfloat k1, jfloat b) {
    (void)env; (void)cls;
    InferBM25 idx = infer_bm25_create((float)k1, (float)b);
    return HANDLE(idx);
}

JNIEXPORT void JNICALL
Java_com_infergo_InfergoNative_bm25Insert(JNIEnv *env, jclass cls,
                                           jlong handle, jlong id, jstring jtext) {
    (void)cls;
    const char *text = (*env)->GetStringUTFChars(env, jtext, NULL);
    if (!text) return;
    infer_bm25_insert(PTR(handle), (int64_t)id, text);
    (*env)->ReleaseStringUTFChars(env, jtext, text);
}

JNIEXPORT void JNICALL
Java_com_infergo_InfergoNative_bm25Remove(JNIEnv *env, jclass cls,
                                           jlong handle, jlong id) {
    (void)env; (void)cls;
    infer_bm25_remove(PTR(handle), (int64_t)id);
}

JNIEXPORT jlongArray JNICALL
Java_com_infergo_InfergoNative_bm25SearchIds(JNIEnv *env, jclass cls,
                                              jlong handle, jstring jquery,
                                              jint k, jint maxResults) {
    (void)cls;
    const char *query = (*env)->GetStringUTFChars(env, jquery, NULL);
    if (!query) return NULL;

    int64_t *ids = (int64_t *)malloc(sizeof(int64_t) * (size_t)maxResults);
    float *scores = (float *)malloc(sizeof(float) * (size_t)maxResults);
    if (!ids || !scores) {
        free(ids); free(scores);
        (*env)->ReleaseStringUTFChars(env, jquery, query);
        return NULL;
    }

    int n = infer_bm25_search(PTR(handle), query, (int)k, ids, scores, (int)maxResults);
    (*env)->ReleaseStringUTFChars(env, jquery, query);

    if (n < 0) {
        free(ids); free(scores);
        return NULL;
    }

    jlongArray result = (*env)->NewLongArray(env, n);
    if (result) {
        (*env)->SetLongArrayRegion(env, result, 0, n, (jlong *)ids);
    }
    free(ids); free(scores);
    return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_infergo_InfergoNative_bm25SearchScores(JNIEnv *env, jclass cls,
                                                 jlong handle, jstring jquery,
                                                 jint k, jint maxResults) {
    (void)cls;
    const char *query = (*env)->GetStringUTFChars(env, jquery, NULL);
    if (!query) return NULL;

    int64_t *ids = (int64_t *)malloc(sizeof(int64_t) * (size_t)maxResults);
    float *scores = (float *)malloc(sizeof(float) * (size_t)maxResults);
    if (!ids || !scores) {
        free(ids); free(scores);
        (*env)->ReleaseStringUTFChars(env, jquery, query);
        return NULL;
    }

    int n = infer_bm25_search(PTR(handle), query, (int)k, ids, scores, (int)maxResults);
    (*env)->ReleaseStringUTFChars(env, jquery, query);

    if (n < 0) {
        free(ids); free(scores);
        return NULL;
    }

    jfloatArray result = (*env)->NewFloatArray(env, n);
    if (result) {
        (*env)->SetFloatArrayRegion(env, result, 0, n, scores);
    }
    free(ids); free(scores);
    return result;
}

JNIEXPORT jint JNICALL
Java_com_infergo_InfergoNative_bm25Size(JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    return (jint)infer_bm25_size(PTR(handle));
}

JNIEXPORT jint JNICALL
Java_com_infergo_InfergoNative_bm25Save(JNIEnv *env, jclass cls,
                                         jlong handle, jstring jpath) {
    (void)cls;
    const char *path = (*env)->GetStringUTFChars(env, jpath, NULL);
    if (!path) return -1;
    int rc = infer_bm25_save(PTR(handle), path);
    (*env)->ReleaseStringUTFChars(env, jpath, path);
    return (jint)rc;
}

JNIEXPORT jint JNICALL
Java_com_infergo_InfergoNative_bm25Load(JNIEnv *env, jclass cls,
                                         jlong handle, jstring jpath) {
    (void)cls;
    const char *path = (*env)->GetStringUTFChars(env, jpath, NULL);
    if (!path) return -1;
    int rc = infer_bm25_load(PTR(handle), path);
    (*env)->ReleaseStringUTFChars(env, jpath, path);
    return (jint)rc;
}

JNIEXPORT void JNICALL
Java_com_infergo_InfergoNative_bm25Free(JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    infer_bm25_free(PTR(handle));
}
