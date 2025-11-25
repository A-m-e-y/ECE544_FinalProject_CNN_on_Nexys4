#include "reduced_cnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dataset_arrays.h"

// Local forward helpers (duplicate from reduced_cnn.c to avoid refactoring)
static void conv3x3_same_local(const float *in, int in_ch, int out_ch,
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
                float acc = 0.0f;
                for (int ic = 0; ic < in_ch; ++ic)
                {
                    const float *wbase = weights + (oc * in_ch + ic) * 3 * 3;
                    for (int ky = 0; ky < 3; ++ky)
                    {
                        int iy = y + ky - 1;
                        if (iy < 0 || iy >= H)
                            continue;
                        for (int kx = 0; kx < 3; ++kx)
                        {
                            int ix = x + kx - 1;
                            if (ix < 0 || ix >= W)
                                continue;
                            float v = in[(ic * H + iy) * W + ix];
                            float w = wbase[ky * 3 + kx];
                            acc += v * w;
                        }
                    }
                }
                acc += biases[oc];
                if (acc < 0)
                    acc = 0; // ReLU
                out[(oc * H + y) * W + x] = acc;
            }
        }
    }
}

static void conv1x1_local(const float *in, int in_ch, int out_ch,
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
                    float w = weights[oc * in_ch + ic];
                    acc += v * w;
                }
                out[(oc * H + y) * W + x] = acc;
            }
        }
    }
}

static void global_avg_pool_local(const float *feat, int ch, float *logits)
{
    const int H = IMG_SIZE, W = IMG_SIZE;
    int HW = H * W;
    for (int c = 0; c < ch; ++c)
    {
        double sum = 0.0;
        const float *base = feat + c * HW;
        for (int i = 0; i < HW; ++i)
            sum += base[i];
        logits[c] = (float)(sum / HW);
    }
}

static void softmax_local(float *v, int n)
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

float cross_entropy_loss(const float *probs, int label)
{
    float p = probs[label];
    if (p < 1e-8f)
        p = 1e-8f;
    return -logf(p);
}

static void backward_conv1x1_local(const float *in, const float *grad_out,
                                   int in_ch, int out_ch,
                                   const float *weights,
                                   float *grad_in, float *grad_w, float *grad_b)
{
    const int H = IMG_SIZE, W = IMG_SIZE;
    memset(grad_in, 0, sizeof(float) * in_ch * H * W);
    for (int oc = 0; oc < out_ch; ++oc)
    {
        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                float go = grad_out[(oc * H + y) * W + x];
                grad_b[oc] += go;
                for (int ic = 0; ic < in_ch; ++ic)
                {
                    float v = in[(ic * H + y) * W + x];
                    grad_w[oc * in_ch + ic] += v * go;
                    grad_in[(ic * H + y) * W + x] += weights[oc * in_ch + ic] * go;
                }
            }
        }
    }
}

static void backward_conv3x3_relu_local(const float *in, const float *out, const float *grad_out,
                                        int in_ch, int out_ch,
                                        const float *weights,
                                        float *grad_in, float *grad_w, float *grad_b)
{
    const int H = IMG_SIZE, W = IMG_SIZE;
    memset(grad_in, 0, sizeof(float) * in_ch * H * W);
    for (int oc = 0; oc < out_ch; ++oc)
    {
        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                float go = grad_out[(oc * H + y) * W + x];
                if (out[(oc * H + y) * W + x] <= 0.0f)
                    go = 0.0f; // ReLU mask
                grad_b[oc] += go;
                for (int ic = 0; ic < in_ch; ++ic)
                {
                    const float *wbase = weights + (oc * in_ch + ic) * 3 * 3;
                    for (int ky = 0; ky < 3; ++ky)
                    {
                        int iy = y + ky - 1;
                        if (iy < 0 || iy >= H)
                            continue;
                        for (int kx = 0; kx < 3; ++kx)
                        {
                            int ix = x + kx - 1;
                            if (ix < 0 || ix >= W)
                                continue;
                            float inval = in[(ic * H + iy) * W + ix];
                            grad_w[(oc * in_ch + ic) * 3 * 3 + ky * 3 + kx] += inval * go;
                            grad_in[(ic * H + iy) * W + ix] += wbase[ky * 3 + kx] * go;
                        }
                    }
                }
            }
        }
    }
}

int train_reduced_cnn_embedded(ReducedCNNWeights *w, int epochs, float lr)
{
    int num_samples = DATASET_NUM_SAMPLES;
    int num_classes = DATASET_NUM_CLASSES;
    if (num_samples == 0)
    {
        fprintf(stderr, "No samples in embedded dataset.\n");
        return -1;
    }
    // Buffers
    float *act1 = (float *)malloc(CONV1_OUT_CH * IMG_SIZE * IMG_SIZE * sizeof(float));
    float *act2 = (float *)malloc(CONV2_OUT_CH * IMG_SIZE * IMG_SIZE * sizeof(float));
    float *act3 = (float *)malloc(CONV3_OUT_CH * IMG_SIZE * IMG_SIZE * sizeof(float));
    float *act_cls = (float *)malloc(CONVCLS_OUT_CH * IMG_SIZE * IMG_SIZE * sizeof(float));
    float *input = (float *)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
    float probs[NUM_CLASSES];
    if (!act1 || !act2 || !act3 || !act_cls || !input)
    {
        fprintf(stderr, "Alloc fail\n");
        return -2;
    }

    // Grad buffers
    float *g_convcls_w = (float *)calloc(CONVCLS_WEIGHTS, sizeof(float));
    float *g_convcls_b = (float *)calloc(CONVCLS_OUT_CH, sizeof(float));
    float *g_conv3_w = (float *)calloc(CONV3_WEIGHTS, sizeof(float));
    float *g_conv3_b = (float *)calloc(CONV3_OUT_CH, sizeof(float));
    float *g_conv2_w = (float *)calloc(CONV2_WEIGHTS, sizeof(float));
    float *g_conv2_b = (float *)calloc(CONV2_OUT_CH, sizeof(float));
    float *g_conv1_w = (float *)calloc(CONV1_WEIGHTS, sizeof(float));
    float *g_conv1_b = (float *)calloc(CONV1_OUT_CH, sizeof(float));
    float *g_act3 = (float *)malloc(CONV3_OUT_CH * IMG_SIZE * IMG_SIZE * sizeof(float));
    float *g_act2 = (float *)malloc(CONV2_OUT_CH * IMG_SIZE * IMG_SIZE * sizeof(float));
    float *g_act1 = (float *)malloc(CONV1_OUT_CH * IMG_SIZE * IMG_SIZE * sizeof(float));
    float *g_input = (float *)malloc(CONV1_IN_CH * IMG_SIZE * IMG_SIZE * sizeof(float));
    float *g_act_cls = (float *)malloc(CONVCLS_OUT_CH * IMG_SIZE * IMG_SIZE * sizeof(float));
    if (!g_convcls_w || !g_convcls_b || !g_conv3_w || !g_conv3_b || !g_conv2_w || !g_conv2_b || !g_conv1_w || !g_conv1_b || !g_act3 || !g_act2 || !g_act1 || !g_input || !g_act_cls)
    {
        fprintf(stderr, "Grad alloc fail\n");
        return -3;
    }

    // Index array for shuffling to avoid order bias
    int *indices = (int *)malloc(sizeof(int) * num_samples);
    for (int i = 0; i < num_samples; ++i)
        indices[i] = i;
    unsigned rng_state = 0xC0FFEEu; // simple deterministic seed

    for (int ep = 0; ep < epochs; ++ep)
    {
        // Fisher-Yates shuffle
        for (int i = num_samples - 1; i > 0; --i)
        {
            // xorshift32 for pseudo-random
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            unsigned r = rng_state % (unsigned)(i + 1);
            int tmp = indices[i];
            indices[i] = indices[r];
            indices[r] = tmp;
        }
        double total_loss = 0.0;
        int correct = 0;
        for (int si = 0; si < num_samples; ++si)
        {
            int s = indices[si];
            // Convert bits to float input
            const uint32_t *bits = training_samples[s].data;
            for (int i = 0; i < IMG_SIZE * IMG_SIZE; ++i)
            {
                union
                {
                    uint32_t u;
                    float f;
                } cvt;
                cvt.u = bits[i];
                input[i] = cvt.f;
            }
            int label = training_samples[s].label;
            // Forward
            conv3x3_same_local(input, CONV1_IN_CH, CONV1_OUT_CH, w->conv1_w, w->conv1_b, act1);
            conv3x3_same_local(act1, CONV2_IN_CH, CONV2_OUT_CH, w->conv2_w, w->conv2_b, act2);
            conv3x3_same_local(act2, CONV3_IN_CH, CONV3_OUT_CH, w->conv3_w, w->conv3_b, act3);
            conv1x1_local(act3, CONVCLS_IN_CH, CONVCLS_OUT_CH, w->convcls_w, w->convcls_b, act_cls);
            global_avg_pool_local(act_cls, CONVCLS_OUT_CH, probs);
            softmax_local(probs, CONVCLS_OUT_CH);
            float loss = cross_entropy_loss(probs, label);
            total_loss += loss;
            // Prediction
            int pred = 0;
            float bestp = probs[0];
            for (int c = 1; c < num_classes; ++c)
            {
                if (probs[c] > bestp)
                {
                    bestp = probs[c];
                    pred = c;
                }
            }
            if (pred == label)
                ++correct;
            // Zero grads
            memset(g_convcls_w, 0, CONVCLS_WEIGHTS * sizeof(float));
            memset(g_convcls_b, 0, CONVCLS_OUT_CH * sizeof(float));
            memset(g_conv3_w, 0, CONV3_WEIGHTS * sizeof(float));
            memset(g_conv3_b, 0, CONV3_OUT_CH * sizeof(float));
            memset(g_conv2_w, 0, CONV2_WEIGHTS * sizeof(float));
            memset(g_conv2_b, 0, CONV2_OUT_CH * sizeof(float));
            memset(g_conv1_w, 0, CONV1_WEIGHTS * sizeof(float));
            memset(g_conv1_b, 0, CONV1_OUT_CH * sizeof(float));
            // dSoftmax+CE
            float g_logits[CONVCLS_OUT_CH];
            for (int c = 0; c < CONVCLS_OUT_CH; ++c)
                g_logits[c] = probs[c];
            if (label < CONVCLS_OUT_CH)
                g_logits[label] -= 1.0f;
            // Spread over spatial via GAP
            int HW = IMG_SIZE * IMG_SIZE;
            for (int oc = 0; oc < CONVCLS_OUT_CH; ++oc)
            {
                float gshare = g_logits[oc] / (float)HW;
                for (int i = 0; i < HW; ++i)
                    g_act_cls[oc * HW + i] = gshare;
            }
            // Backward layers
            backward_conv1x1_local(act3, g_act_cls, CONVCLS_IN_CH, CONVCLS_OUT_CH, w->convcls_w, g_act3, g_convcls_w, g_convcls_b);
            backward_conv3x3_relu_local(act2, act3, g_act3, CONV3_IN_CH, CONV3_OUT_CH, w->conv3_w, g_act2, g_conv3_w, g_conv3_b);
            backward_conv3x3_relu_local(act1, act2, g_act2, CONV2_IN_CH, CONV2_OUT_CH, w->conv2_w, g_act1, g_conv2_w, g_conv2_b);
            backward_conv3x3_relu_local(input, act1, g_act1, CONV1_IN_CH, CONV1_OUT_CH, w->conv1_w, g_input, g_conv1_w, g_conv1_b);
            // SGD update
            for (size_t i = 0; i < CONVCLS_WEIGHTS; ++i)
                w->convcls_w[i] -= lr * g_convcls_w[i];
            for (int i = 0; i < CONVCLS_OUT_CH; ++i)
                w->convcls_b[i] -= lr * g_convcls_b[i];
            for (size_t i = 0; i < CONV3_WEIGHTS; ++i)
                w->conv3_w[i] -= lr * g_conv3_w[i];
            for (int i = 0; i < CONV3_OUT_CH; ++i)
                w->conv3_b[i] -= lr * g_conv3_b[i];
            for (size_t i = 0; i < CONV2_WEIGHTS; ++i)
                w->conv2_w[i] -= lr * g_conv2_w[i];
            for (int i = 0; i < CONV2_OUT_CH; ++i)
                w->conv2_b[i] -= lr * g_conv2_b[i];
            for (size_t i = 0; i < CONV1_WEIGHTS; ++i)
                w->conv1_w[i] -= lr * g_conv1_w[i];
            for (int i = 0; i < CONV1_OUT_CH; ++i)
                w->conv1_b[i] -= lr * g_conv1_b[i];
        }
        float avg_loss = (float)(total_loss / num_samples);
        float acc = (float)correct / (float)num_samples;
        printf("[Train] Epoch %d/%d AvgLoss=%.6f Acc=%.4f\n", ep + 1, epochs, avg_loss, acc);
    }
    free(indices);
    free(act1);
    free(act2);
    free(act3);
    free(act_cls);
    free(input);
    free(g_convcls_w);
    free(g_convcls_b);
    free(g_conv3_w);
    free(g_conv3_b);
    free(g_conv2_w);
    free(g_conv2_b);
    free(g_conv1_w);
    free(g_conv1_b);
    free(g_act3);
    free(g_act2);
    free(g_act1);
    free(g_input);
    free(g_act_cls);
    return 0;
}
