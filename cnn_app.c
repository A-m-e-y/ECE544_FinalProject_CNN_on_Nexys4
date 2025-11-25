#include "reduced_cnn.h"
#include "dataset_arrays.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static void usage()
{
    printf("Usage:\n  ./cnn_app train <epochs> <lr>\n  ./cnn_app infer <sample_index>\n");
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        usage();
        return 1;
    }
    ReducedCNNWeights w;
    if (strcmp(argv[1], "train") == 0)
    {
        int epochs = (argc >= 3) ? atoi(argv[2]) : 1;
        float lr = (argc >= 4) ? (float)atof(argv[3]) : 0.01f;
        reduced_cnn_init_random(&w, 1234);
        printf("Starting training: samples=%d classes=%d epochs=%d lr=%.4f\n", DATASET_NUM_SAMPLES, DATASET_NUM_CLASSES, epochs, lr);
        if (train_reduced_cnn_embedded(&w, epochs, lr) != 0)
        {
            fprintf(stderr, "Training failed\n");
            return 2;
        }
        if (reduced_cnn_save_weights(&w, "trained_reduced_cnn.bin") != 0)
        {
            fprintf(stderr, "Save failed\n");
            return 3;
        }
        printf("Weights saved to trained_reduced_cnn.bin\n");
        return 0;
    }
    else if (strcmp(argv[1], "infer") == 0)
    {
        if (reduced_cnn_load_weights(&w, "trained_reduced_cnn.bin") != 0)
        {
            fprintf(stderr, "Load weights failed\n");
            return 4;
        }
        int index = (argc >= 3) ? atoi(argv[2]) : 0;
        if (index < 0 || index >= DATASET_NUM_SAMPLES)
        {
            fprintf(stderr, "Invalid sample index\n");
            return 5;
        }
        const uint32_t *bits = training_samples[index].data;
        float input[IMG_SIZE * IMG_SIZE];
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
        float probs[NUM_CLASSES];
        reduced_cnn_forward(&w, input, probs);
        int best = 0;
        float bestv = probs[0];
        for (int c = 0; c < NUM_CLASSES; ++c)
        {
            printf("Class %d: %.6f\n", c, probs[c]);
            if (probs[c] > bestv)
            {
                bestv = probs[c];
                best = c;
            }
        }
        printf("Predicted: %d (%.6f) label=%d\n", best, bestv, training_samples[index].label);
        return 0;
    }
    else
    {
        usage();
        return 1;
    }
}
