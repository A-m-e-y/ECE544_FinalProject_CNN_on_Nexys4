// reduced_cnn.h
// Lightweight inference-only CNN without dense layers.
// Architecture:
//   conv1: 1 -> 8, 3x3, pad=1, stride=1, ReLU
//   conv2: 8 -> 16, 3x3, pad=1, stride=1, ReLU
//   conv3: 16 -> 32, 3x3, pad=1, stride=1, ReLU
//   conv_cls: 32 -> 10, 1x1, stride=1 (classification logits)
//   global average pooling over spatial (10x10) -> 10 logits
//   softmax
// All weights are stored as contiguous float arrays flattened per filter.
// This design avoids very large K sizes and dense layers.

#ifndef REDUCED_CNN_H
#define REDUCED_CNN_H

#include <stddef.h>

#define IMG_SIZE 10
#define NUM_CLASSES 10

#define CONV1_IN_CH 1
#define CONV1_OUT_CH 8
#define CONV2_IN_CH CONV1_OUT_CH
#define CONV2_OUT_CH 16
#define CONV3_IN_CH CONV2_OUT_CH
#define CONV3_OUT_CH 32
#define CONVCLS_IN_CH CONV3_OUT_CH
#define CONVCLS_OUT_CH NUM_CLASSES

// Kernel sizes
#define K3 3
#define K1 1

// Weight array sizes (filters stored as [out][in][kH][kW]) flattened
#define CONV1_WEIGHTS (CONV1_OUT_CH * CONV1_IN_CH * K3 * K3)
#define CONV2_WEIGHTS (CONV2_OUT_CH * CONV2_IN_CH * K3 * K3)
#define CONV3_WEIGHTS (CONV3_OUT_CH * CONV3_IN_CH * K3 * K3)
#define CONVCLS_WEIGHTS (CONVCLS_OUT_CH * CONVCLS_IN_CH * K1 * K1)

typedef struct
{
    float conv1_w[CONV1_WEIGHTS];
    float conv1_b[CONV1_OUT_CH];
    float conv2_w[CONV2_WEIGHTS];
    float conv2_b[CONV2_OUT_CH];
    float conv3_w[CONV3_WEIGHTS];
    float conv3_b[CONV3_OUT_CH];
    float convcls_w[CONVCLS_WEIGHTS];
    float convcls_b[CONVCLS_OUT_CH];
} ReducedCNNWeights;

// Forward (inference) API
// input: (1 x 1 x IMG_SIZE x IMG_SIZE) grayscale normalized [0,1]
// output: softmax probabilities length NUM_CLASSES
void reduced_cnn_init_random(ReducedCNNWeights *w, unsigned seed);
void reduced_cnn_forward(const ReducedCNNWeights *w, const float *input, float *probs);

// Training support -----------------------------------------------------------
// Scan dataset directory structure root/CLASS/*.jpg returning file paths and labels.
// Automatically detects number of classes present (0..9 subdirectories).
// Returns number of samples loaded or -1 on error.
int scan_dataset(const char *root, char ***out_paths, int **out_labels, int *out_num_classes);

// Train the reduced CNN for given epochs (SGD batch size 1) with learning rate.
// paths: array of image path strings, labels: integer labels in [0,num_classes-1]
// num_samples: size of arrays, num_classes: discovered classes.
// Returns 0 on success.
int train_reduced_cnn(ReducedCNNWeights *w,
                      char **paths, int *labels,
                      int num_samples, int num_classes,
                      int epochs, float lr);

// Load single grayscale image (JPG/PNG) and resize to IMG_SIZE x IMG_SIZE, normalize 0..1.
int load_image_to_tensor(const char *path, float *tensor_out); // tensor_out size IMG_SIZE*IMG_SIZE

// Utility: load/save weights from simple binary file layout (sequential floats)
// Expects order matching struct field order above.
int reduced_cnn_load_weights(ReducedCNNWeights *w, const char *path);
int reduced_cnn_save_weights(const ReducedCNNWeights *w, const char *path);

// Cross entropy loss for single sample; label is integer class.
float cross_entropy_loss(const float *probs, int label);

// Embedded header-based training using generated dataset_arrays.h (training_samples[])
int train_reduced_cnn_embedded(ReducedCNNWeights *w, int epochs, float lr);

#endif // REDUCED_CNN_H
