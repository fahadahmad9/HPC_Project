#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9
#define BLOCK_SIZE 256  // CUDA block size

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Activation functions on CPU (kept for reference)
void relu(double* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax(double* x, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// CUDA kernel for matrix-vector multiplication
__global__ void matrixVectorMulKernel(double* d_matrix, double* d_input, double* d_output, 
                                      double* d_bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < rows) {
        double sum = d_bias[idx];
        for (int j = 0; j < cols; j++) {
            sum += d_matrix[idx * cols + j] * d_input[j];
        }
        d_output[idx] = sum;
    }
}

// CUDA kernel for ReLU activation
__global__ void reluKernel(double* d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        d_data[idx] = (d_data[idx] > 0) ? d_data[idx] : 0;
    }
}

// CUDA kernel for softmax activation (two-phase approach)
__global__ void softmaxExpKernel(double* d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        d_data[idx] = exp(d_data[idx]);
    }
}

__global__ void softmaxNormalizeKernel(double* d_data, double sum, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        d_data[idx] /= sum;
    }
}

// CUDA kernel for output gradient calculation
__global__ void outputGradientKernel(double* d_output, double* d_target, double* d_gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        d_gradient[idx] = d_output[idx] - d_target[idx];
    }
}

// CUDA kernel for hidden gradient calculation
__global__ void hiddenGradientKernel(double* d_W2, double* d_output_gradient, 
                                     double* d_hidden, double* d_hidden_gradient, 
                                     int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < hidden_size) {
        double sum = 0.0;
        for (int j = 0; j < output_size; j++) {
            sum += d_W2[j * hidden_size + idx] * d_output_gradient[j];
        }
        // ReLU derivative
        d_hidden_gradient[idx] = sum * (d_hidden[idx] > 0 ? 1.0 : 0.0);
    }
}

// CUDA kernel for weight update
__global__ void updateWeightsKernel(double* d_weights, double* d_gradient, double* d_inputs,
                                   int rows, int cols, double learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / cols;
    int col = idx % cols;
    
    if (row < rows && col < cols) {
        d_weights[row * cols + col] -= learning_rate * d_gradient[row] * d_inputs[col];
    }
}

// CUDA kernel for bias update
__global__ void updateBiasKernel(double* d_bias, double* d_gradient, 
                                int size, double learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        d_bias[idx] -= learning_rate * d_gradient[idx];
    }
}

// Neural network structure
typedef struct {
    // Host matrices
    double** W1;
    double** W2;
    double* b1;
    double* b2;
    
    // Device matrices (flattened for CUDA)
    double* d_W1;
    double* d_W2;
    double* d_b1;
    double* d_b2;
    double* d_input;
    double* d_hidden;
    double* d_output;
    double* d_target;
    double* d_hidden_gradient;
    double* d_output_gradient;
} NeuralNetwork;

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Allocate host memory
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    // Initialize weights with random values
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_b1, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_b2, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_input, INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_output, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_target, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_hidden_gradient, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_output_gradient, OUTPUT_SIZE * sizeof(double)));
    
    // Copy initial weights to device
    double* W1_flat = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* W2_flat = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    
    // Flatten W1
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            W1_flat[i * INPUT_SIZE + j] = net->W1[i][j];
        }
    }
    
    // Flatten W2
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            W2_flat[i * HIDDEN_SIZE + j] = net->W2[i][j];
        }
    }
    
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W1, W1_flat, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W2, W2_flat, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    free(W1_flat);
    free(W2_flat);
    
    return net;
}

// Forward pass using CUDA
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    // Calculate hidden layer values: hidden = W1 * input + b1
    dim3 hiddenBlocks((HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixVectorMulKernel<<<hiddenBlocks, BLOCK_SIZE>>>(net->d_W1, net->d_input, net->d_hidden, 
                                                       net->d_b1, HIDDEN_SIZE, INPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Apply ReLU activation to hidden layer
    reluKernel<<<hiddenBlocks, BLOCK_SIZE>>>(net->d_hidden, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Calculate output layer values: output = W2 * hidden + b2
    dim3 outputBlocks((OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixVectorMulKernel<<<outputBlocks, BLOCK_SIZE>>>(net->d_W2, net->d_hidden, net->d_output, 
                                                       net->d_b2, OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Apply softmax to output layer
    softmaxExpKernel<<<outputBlocks, BLOCK_SIZE>>>(net->d_output, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Calculate sum for softmax (reduction)
    double* output_host = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    CHECK_CUDA_ERROR(cudaMemcpy(output_host, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    
    double sum = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        sum += output_host[i];
    }
    
    // Normalize softmax values
    softmaxNormalizeKernel<<<outputBlocks, BLOCK_SIZE>>>(net->d_output, sum, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(hidden, net->d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(output, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    
    free(output_host);
}

// Backpropagation using CUDA
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    // Copy target to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_target, target, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    // Make sure hidden and output are on device (in case forward wasn't just called)
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_hidden, hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_output, output, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    // Calculate output layer gradient: d_output = output - target
    dim3 outputBlocks((OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    outputGradientKernel<<<outputBlocks, BLOCK_SIZE>>>(net->d_output, net->d_target, 
                                                     net->d_output_gradient, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Calculate hidden layer gradient: d_hidden = W2.T * d_output * relu_derivative(hidden)
    dim3 hiddenBlocks((HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    hiddenGradientKernel<<<hiddenBlocks, BLOCK_SIZE>>>(net->d_W2, net->d_output_gradient, 
                                                     net->d_hidden, net->d_hidden_gradient, 
                                                     HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Update W2: W2 -= learning_rate * d_output * hidden.T
    dim3 w2Blocks((OUTPUT_SIZE * HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    updateWeightsKernel<<<w2Blocks, BLOCK_SIZE>>>(net->d_W2, net->d_output_gradient, net->d_hidden, 
                                                OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Update W1: W1 -= learning_rate * d_hidden * input.T
    dim3 w1Blocks((HIDDEN_SIZE * INPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    updateWeightsKernel<<<w1Blocks, BLOCK_SIZE>>>(net->d_W1, net->d_hidden_gradient, net->d_input, 
                                                HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Update b2: b2 -= learning_rate * d_output
    updateBiasKernel<<<outputBlocks, BLOCK_SIZE>>>(net->d_b2, net->d_output_gradient, 
                                                 OUTPUT_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Update b1: b1 -= learning_rate * d_hidden
    updateBiasKernel<<<hiddenBlocks, BLOCK_SIZE>>>(net->d_b1, net->d_hidden_gradient, 
                                                 HIDDEN_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Sync after all kernels (to ensure they finish before we exit the function)
    cudaDeviceSynchronize();
    
    // Update host weights from device at the end of training
    double* W1_flat = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* W2_flat = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    
    // Copy from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(W1_flat, net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(W2_flat, net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(net->b1, net->d_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(net->b2, net->d_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Unflatten matrices
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->W1[i][j] = W1_flat[i * INPUT_SIZE + j];
        }
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->W2[i][j] = W2_flat[i * HIDDEN_SIZE + j];
        }
    }
    
    free(W1_flat);
    free(W2_flat);
}

// Train network (same as before)
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            forward(net, images[i], hidden, output);
            backward(net, images[i], hidden, output, labels[i]);

            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
}

// Free network memory with CUDA cleanup
void freeNetwork(NeuralNetwork* net) {
    // Free device memory
    cudaFree(net->d_W1);
    cudaFree(net->d_W2);
    cudaFree(net->d_b1);
    cudaFree(net->d_b2);
    cudaFree(net->d_input);
    cudaFree(net->d_hidden);
    cudaFree(net->d_output);
    cudaFree(net->d_target);
    cudaFree(net->d_hidden_gradient);
    cudaFree(net->d_output_gradient);
    
    // Free host memory
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
}

// Read MNIST dataset (same as before)
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

// Evaluate accuracy on test data (same as before)
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forward(net, images[i], hidden, output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// Main function (same as before)
int main() {
    printf("MNIST Neural Network with CUDA\n\n");

    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);

    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    freeNetwork(net);
    
    // Free dataset memory
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);
    
    // Reset CUDA device
    cudaDeviceReset();
    
    return 0;
}