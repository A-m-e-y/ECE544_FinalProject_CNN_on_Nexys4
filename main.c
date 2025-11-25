// main.c
// Test harness for reduced CNN pure C implementation.

#include "reduced_cnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    ReducedCNNWeights w;
    if (argc == 2 && strcmp(argv[1], "load") == 0)
    {
        if (reduced_cnn_load_weights(&w, "reduced_weights.bin") != 0)
        {
            fprintf(stderr, "Failed to load weights, initializing random.\n");
            reduced_cnn_init_random(&w, 1234);
        }
    }
    else
    {
        reduced_cnn_init_random(&w, 1234);
    }

    // Prepare dummy normalized input (simple pattern)
    float input[CONV1_IN_CH * IMG_SIZE * IMG_SIZE];
    for (int y = 0; y < IMG_SIZE; ++y)
    {
        for (int x = 0; x < IMG_SIZE; ++x)
        {
            input[y * IMG_SIZE + x] = (float)((x + y) % 10) / 9.0f; // range [0,1]
        }
    }

    float probs[NUM_CLASSES];
    reduced_cnn_forward(&w, input, probs);

    // Print probabilities and predicted class
    int best = 0;
    float bestv = probs[0];
    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        printf("Class %d: %.6f\n", i, probs[i]);
        if (probs[i] > bestv)
        {
            bestv = probs[i];
            best = i;
        }
    }
    printf("Predicted: %d (%.6f)\n", best, bestv);

    // Optionally save weights
    if (argc == 2 && strcmp(argv[1], "save") == 0)
    {
        if (reduced_cnn_save_weights(&w, "reduced_weights.bin") == 0)
        {
            printf("Weights saved to reduced_weights.bin\n");
        }
    }
    return 0;
}
