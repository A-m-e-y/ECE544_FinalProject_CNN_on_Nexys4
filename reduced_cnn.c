// reduced_cnn.c
// Pure C implementation of reduced CNN forward pass.

#include "reduced_cnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static float frand(unsigned *state)
{
    // xorshift32
    unsigned x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return (x & 0xFFFFFF) / (float)0x1000000 - 0.5f; // approx [-0.5,0.5)
}

void reduced_cnn_init_random(ReducedCNNWeights *w, unsigned seed)
{
    unsigned st = seed ? seed : 1u;
    for (size_t i = 0; i < CONV1_WEIGHTS; i++)
        w->conv1_w[i] = frand(&st) * 0.1f;
    for (size_t i = 0; i < CONV1_OUT_CH; i++)
        w->conv1_b[i] = 0.0f;
    for (size_t i = 0; i < CONV2_WEIGHTS; i++)
        w->conv2_w[i] = frand(&st) * 0.1f;
    for (size_t i = 0; i < CONV2_OUT_CH; i++)
        w->conv2_b[i] = 0.0f;
    for (size_t i = 0; i < CONV3_WEIGHTS; i++)
        w->conv3_w[i] = frand(&st) * 0.1f;
    for (size_t i = 0; i < CONV3_OUT_CH; i++)
        w->conv3_b[i] = 0.0f;
    for (size_t i = 0; i < CONVCLS_WEIGHTS; i++)
        w->convcls_w[i] = frand(&st) * 0.1f;
    for (size_t i = 0; i < CONVCLS_OUT_CH; i++)
        w->convcls_b[i] = 0.0f;
}

int reduced_cnn_load_weights(ReducedCNNWeights *w, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    size_t need = sizeof(*w); // contiguous layout guaranteed
    size_t rd = fread(w, 1, need, f);
    fclose(f);
    return rd == need ? 0 : -2;
}

int reduced_cnn_save_weights(const ReducedCNNWeights *w, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    size_t need = sizeof(*w);
    size_t wr = fwrite(w, 1, need, f);
    fclose(f);
    return wr == need ? 0 : -2;
}

// Helper to perform padded 3x3 convolution (same padding) with ReLU.
static void conv3x3_same(const float *in, int in_ch, int out_ch,
                         const float *weights, const float *biases,
                         float *out)
{
    // in: (in_ch, IMG_SIZE, IMG_SIZE) ; out: (out_ch, IMG_SIZE, IMG_SIZE)
    const int H = IMG_SIZE, W = IMG_SIZE;
    for (int oc = 0; oc < out_ch; ++oc)
    {
        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                float acc = 0.0f;
                // kernel loops
                for (int ic = 0; ic < in_ch; ++ic)
                {
                    const float *wbase = weights + (oc * in_ch + ic) * K3 * K3;
                    for (int ky = 0; ky < K3; ++ky)
                    {
                        int iy = y + ky - 1; // pad=1
                        if (iy < 0 || iy >= H)
                            continue;
                        for (int kx = 0; kx < K3; ++kx)
                        {
                            int ix = x + kx - 1;
                            if (ix < 0 || ix >= W)
                                continue;
                            float v = in[(ic * H + iy) * W + ix];
                            float w = wbase[ky * K3 + kx];
                            acc += v * w;
                        }
                    }
                }
                acc += biases[oc];
                // ReLU
                if (acc < 0)
                    acc = 0;
                out[(oc * H + y) * W + x] = acc;
            }
        }
    }
}

// 1x1 conv classification (no padding) no ReLU.
static void conv1x1(const float *in, int in_ch, int out_ch,
                    const float *weights, const float *biases,
                    float *out)
{
    const int H = IMG_SIZE, W = IMG_SIZE;
    for (int oc = 0; oc < out_ch; ++oc)
    {
        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                float acc = biases[oc];
                for (int ic = 0; ic < in_ch; ++ic)
                {
                    float v = in[(ic * H + y) * W + x];
                    float w = weights[(oc * in_ch + ic)]; // 1x1 single weight per in channel
                    acc += v * w;
                }
                out[(oc * H + y) * W + x] = acc;
            }
        }
    }
}

// Global average pooling across spatial dims, returning logits vector
static void global_avg_pool(const float *feat, int ch, float *logits)
{
    const int H = IMG_SIZE, W = IMG_SIZE;
    int HW = H * W;
    for (int c = 0; c < ch; ++c)
    {
        double sum = 0.0; // use double for accumulation accuracy
        const float *base = feat + c * H * W;
        for (int i = 0; i < HW; ++i)
            sum += base[i];
        logits[c] = (float)(sum / HW);
    }
}

static void softmax(float *v, int n)
{
    float maxv = v[0];
    for (int i = 1; i < n; ++i)
        if (v[i] > maxv)
            maxv = v[i];
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
    {
        v[i] = (float)exp((double)v[i] - maxv);
        sum += v[i];
    }
    for (int i = 0; i < n; ++i)
        v[i] = (float)(v[i] / sum);
}

void reduced_cnn_forward(const ReducedCNNWeights *w, const float *input, float *probs)
{
    // Buffers: we reuse memory to limit footprint.
    static float buf1[CONV1_OUT_CH * IMG_SIZE * IMG_SIZE];
    static float buf2[CONV2_OUT_CH * IMG_SIZE * IMG_SIZE];
    static float buf3[CONV3_OUT_CH * IMG_SIZE * IMG_SIZE];
    static float buf4[CONVCLS_OUT_CH * IMG_SIZE * IMG_SIZE];

    conv3x3_same(input, CONV1_IN_CH, CONV1_OUT_CH, w->conv1_w, w->conv1_b, buf1);
    conv3x3_same(buf1, CONV2_IN_CH, CONV2_OUT_CH, w->conv2_w, w->conv2_b, buf2);
    conv3x3_same(buf2, CONV3_IN_CH, CONV3_OUT_CH, w->conv3_w, w->conv3_b, buf3);
    // classification conv 1x1 (no ReLU before pooling, could add if desired)
    conv1x1(buf3, CONVCLS_IN_CH, CONVCLS_OUT_CH, w->convcls_w, w->convcls_b, buf4);
    // global average pool
    global_avg_pool(buf4, CONVCLS_OUT_CH, probs);
    // softmax
    softmax(probs, CONVCLS_OUT_CH);
}
