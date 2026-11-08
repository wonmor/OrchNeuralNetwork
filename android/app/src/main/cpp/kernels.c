// Matrix kernels for the GPT-2 forward pass. Weights stay in the memory-mapped .tsw file as IEEE
// float16 and are widened to float32 on the fly; activations are float32. Work is split across cores
// with OpenMP. Everything else (layer norm, attention, softmax, GELU) runs in Kotlin.
#include <jni.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

#if defined(__aarch64__)
typedef __fp16 half_t;
static inline float h2f(half_t h) { return (float)h; }
#else
typedef uint16_t half_t;
static float lut[65536];
static int lut_ready = 0;
static float h2f_slow(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff, bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3ff;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float f; memcpy(&f, &bits, 4); return f;
}
static void ensure_lut(void) {
    if (lut_ready) return;
    for (int i = 0; i < 65536; i++) lut[i] = h2f_slow((uint16_t)i);
    lut_ready = 1;
}
static inline float h2f(half_t h) { return lut[h]; }
#endif

#define CHUNK 256

// C[M,N] = A[M,K] x W[K,N] (+ bias[N]). W is row-major float16 (GPT-2 Conv1D layout: [in, out]).
static void matmul(const float *A, int M, int K, const half_t *W, int N, const float *bias, float *C) {
    int chunks = (N + CHUNK - 1) / CHUNK;
    #pragma omp parallel for schedule(static)
    for (int ch = 0; ch < chunks; ch++) {
        int n0 = ch * CHUNK, n1 = n0 + CHUNK < N ? n0 + CHUNK : N, w = n1 - n0;
        float row[CHUNK];
        for (int m = 0; m < M; m++) {
            float *c = C + (size_t)m * N + n0;
            if (bias) memcpy(c, bias + n0, sizeof(float) * w); else memset(c, 0, sizeof(float) * w);
        }
        for (int k = 0; k < K; k++) {
            const half_t *wr = W + (size_t)k * N + n0;
            for (int j = 0; j < w; j++) row[j] = h2f(wr[j]);
            for (int m = 0; m < M; m++) {
                float a = A[(size_t)m * K + k];
                float *c = C + (size_t)m * N + n0;
                for (int j = 0; j < w; j++) c[j] += a * row[j];
            }
        }
    }
}

// out[v] = dot(x[D], E[v, :]) for the tied token-embedding matrix E[V, D].
static void logits(const float *x, const half_t *E, int V, int D, float *out) {
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < V; v++) {
        const half_t *e = E + (size_t)v * D;
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int d = 0;
        for (; d + 4 <= D; d += 4) {
            s0 += x[d] * h2f(e[d]); s1 += x[d + 1] * h2f(e[d + 1]);
            s2 += x[d + 2] * h2f(e[d + 2]); s3 += x[d + 3] * h2f(e[d + 3]);
        }
        for (; d < D; d++) s0 += x[d] * h2f(e[d]);
        out[v] = (s0 + s1) + (s2 + s3);
    }
}

JNIEXPORT void JNICALL
Java_com_johnseong_orchneuralnetwork_engine_NativeKernels_nMatmul(JNIEnv *env, jclass cls, jfloatArray a, jint m, jint k,
        jobject weights, jlong offset, jint n, jfloatArray bias, jfloatArray c) {
#if !defined(__aarch64__)
    ensure_lut();
#endif
    const half_t *W = (const half_t *)((const uint8_t *)(*env)->GetDirectBufferAddress(env, weights) + offset);
    float *A = (*env)->GetPrimitiveArrayCritical(env, a, NULL);
    float *B = bias ? (*env)->GetPrimitiveArrayCritical(env, bias, NULL) : NULL;
    float *C = (*env)->GetPrimitiveArrayCritical(env, c, NULL);
    matmul(A, m, k, W, n, B, C);
    (*env)->ReleasePrimitiveArrayCritical(env, c, C, 0);
    if (B) (*env)->ReleasePrimitiveArrayCritical(env, bias, B, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, a, A, JNI_ABORT);
}

JNIEXPORT void JNICALL
Java_com_johnseong_orchneuralnetwork_engine_NativeKernels_nLogits(JNIEnv *env, jclass cls, jfloatArray x,
        jobject weights, jlong offset, jint v, jint d, jfloatArray out) {
#if !defined(__aarch64__)
    ensure_lut();
#endif
    const half_t *E = (const half_t *)((const uint8_t *)(*env)->GetDirectBufferAddress(env, weights) + offset);
    float *X = (*env)->GetPrimitiveArrayCritical(env, x, NULL);
    float *O = (*env)->GetPrimitiveArrayCritical(env, out, NULL);
    logits(X, E, v, d, O);
    (*env)->ReleasePrimitiveArrayCritical(env, out, O, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, x, X, JNI_ABORT);
}
