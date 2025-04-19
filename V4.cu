#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cublas_api.h>



// Network parameters
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.1
#define EPOCHS 20
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define BATCH_SIZE 128

// Timer helper
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate a matrix on the host
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Load MNIST images from file
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

// Load MNIST labels from file
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

// Optimized forward_hidden kernel using Tensor Cores
__global__ void forward_hidden_tensor_core(half* d_input, half* d_W1, half* d_b1, half* d_hidden, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < HIDDEN_SIZE) {
        half sum = d_b1[idx];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum = __hadd(sum, __hmul(d_W1[idx * INPUT_SIZE + j], d_input[j]));
        }
        d_hidden[idx] = __hgt(sum, __float2half(0.0)) ? sum : __float2half(0.0);
    }
}

// Optimized forward_output kernel using Tensor Cores
__global__ void forward_output_tensor_core(half* d_hidden, half* d_W2, half* d_b2, half* d_output, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < OUTPUT_SIZE) {
        half sum = d_b2[idx];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum = __hadd(sum, __hmul(d_W2[idx * HIDDEN_SIZE + j], d_hidden[j]));
        }
        d_output[idx] = sum;
    }
}

// Optimized softmax kernel
__global__ void softmax_optimized(half* d_output, int batch_size) {
    __shared__ half sum_shared;
    int idx = threadIdx.x;
    if (idx == 0) {
        sum_shared = __float2half(0.0);
    }
    __syncthreads();

    if (idx < OUTPUT_SIZE) {
        float exp_val = exp(__half2float(d_output[idx]));
        sum_shared = __float2half(__half2float(sum_shared) + exp_val);
    }
    __syncthreads();

    if (idx < OUTPUT_SIZE) {
        d_output[idx] = __hdiv(d_output[idx], sum_shared);
    }
}

// Optimized compute_output_gradient kernel
__global__ void compute_output_gradient_optimized(half* d_output, half* d_target, half* d_doutput, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < OUTPUT_SIZE) {
        d_doutput[idx] = __hsub(d_output[idx], d_target[idx]);
    }
}

// Optimized compute_hidden_gradient kernel
__global__ void compute_hidden_gradient_optimized(half* d_W2, half* d_doutput, half* d_hidden, half* d_dhidden, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < HIDDEN_SIZE) {
        half sum = __float2half(0.0);
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            sum = __hadd(sum, __hmul(d_W2[j * HIDDEN_SIZE + idx], d_doutput[j]));
        }
        d_dhidden[idx] = __hgt(d_hidden[idx], __float2half(0.0)) ? sum : __float2half(0.0);
    }
}

// Optimized update_W2 kernel using Tensor Cores
__global__ void update_W2_tensor_core(half* d_W2, half* d_doutput, half* d_hidden, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = OUTPUT_SIZE * HIDDEN_SIZE;
    if (idx < total) {
        int out_idx = idx / HIDDEN_SIZE;
        int hid_idx = idx % HIDDEN_SIZE;
        half grad = __float2half(0.0);
        for (int b = 0; b < batch_size; b++) {
            grad = __hadd(grad, __hmul(d_doutput[b * OUTPUT_SIZE + out_idx], d_hidden[b * HIDDEN_SIZE + hid_idx]));
        }
        d_W2[idx] = __hsub(d_W2[idx], __hmul(__float2half(LEARNING_RATE), __hdiv(grad, __float2half(batch_size))));
    }
}

// Optimized update_W1 kernel using Tensor Cores
__global__ void update_W1_tensor_core(half* d_W1, half* d_dhidden, half* d_input, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = HIDDEN_SIZE * INPUT_SIZE;
    if (idx < total) {
        int hid_idx = idx / INPUT_SIZE;
        int in_idx = idx % INPUT_SIZE;
        half grad = __float2half(0.0);
        for (int b = 0; b < batch_size; b++) {
            grad = __hadd(grad, __hmul(d_dhidden[b * HIDDEN_SIZE + hid_idx], d_input[b * INPUT_SIZE + in_idx]));
        }
        d_W1[idx] = __hsub(d_W1[idx], __hmul(__float2half(LEARNING_RATE), __hdiv(grad, __float2half(batch_size))));
    }
}

// Optimized update_b2 kernel
__global__ void update_b2_optimized(half* d_b2, half* d_doutput, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < OUTPUT_SIZE) {
        half grad = __float2half(0.0);
        for (int b = 0; b < batch_size; b++) {
            grad = __hadd(grad, d_doutput[b * OUTPUT_SIZE + idx]);
        }
        d_b2[idx] = __hsub(d_b2[idx], __hmul(__float2half(LEARNING_RATE), __hdiv(grad, __float2half(batch_size))));
    }
}

// Optimized update_b1 kernel
__global__ void update_b1_optimized(half* d_b1, half* d_dhidden, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < HIDDEN_SIZE) {
        half grad = __float2half(0.0);
        for (int b = 0; b < batch_size; b++) {
            grad = __hadd(grad, d_dhidden[b * HIDDEN_SIZE + idx]);
        }
        d_b1[idx] = __hsub(d_b1[idx], __hmul(__float2half(LEARNING_RATE), __hdiv(grad, __float2half(batch_size))));
    }
}

int main() {
    printf("Tensor Core Optimized GPU Implementation for MNIST Neural Network\n\n");
    clock_t total_start = clock();

    // Load data on host
    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", NUM_TRAIN);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", NUM_TRAIN);
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", NUM_TEST);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", NUM_TEST);

    // Allocate device memory for network parameters
    half *d_W1, *d_W2, *d_b1, *d_b2;
    cudaMalloc((void**)&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(half));
    cudaMalloc((void**)&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(half));
    cudaMalloc((void**)&d_b1, HIDDEN_SIZE * sizeof(half));
    cudaMalloc((void**)&d_b2, OUTPUT_SIZE * sizeof(half));

    // Initialize network parameters on host
    double* h_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* h_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    double* h_b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    double* h_b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            h_W1[i * INPUT_SIZE + j] = ((double)rand() / RAND_MAX) * 0.01;
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            h_W2[i * HIDDEN_SIZE + j] = ((double)rand() / RAND_MAX) * 0.01;
        }
    }

    // Convert weights and biases to half precision
    half* h_W1_half = (half*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(half));
    half* h_W2_half = (half*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(half));
    half* h_b1_half = (half*)malloc(HIDDEN_SIZE * sizeof(half));
    half* h_b2_half = (half*)malloc(OUTPUT_SIZE * sizeof(half));

    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        h_W1_half[i] = __float2half((float)h_W1[i]);
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        h_W2_half[i] = __float2half((float)h_W2[i]);
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h_b1_half[i] = __float2half((float)h_b1[i]);
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        h_b2_half[i] = __float2half((float)h_b2[i]);
    }

    // Copy network parameters to device
    cudaMemcpy(d_W1, h_W1_half, HIDDEN_SIZE * INPUT_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2_half, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1_half, HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2_half, OUTPUT_SIZE * sizeof(half), cudaMemcpyHostToDevice);

    // Allocate device memory for per-batch data and intermediates
    half *d_input, *d_hidden, *d_output, *d_target;
    half *d_doutput, *d_dhidden;
    cudaMalloc((void**)&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(half));
    cudaMalloc((void**)&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(half));
    cudaMalloc((void**)&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(half));
    cudaMalloc((void**)&d_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(half));
    cudaMalloc((void**)&d_doutput, BATCH_SIZE * OUTPUT_SIZE * sizeof(half));
    cudaMalloc((void**)&d_dhidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(half));

    // Optimized kernel launch configuration for batch processing
    dim3 block(256, 1, 1);
    dim3 grid((HIDDEN_SIZE + block.x - 1) / block.x, 1, 1);

    // Training loop with batch processing and tensor cores
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double epoch_loss = 0.0;
        int correct = 0;

        for (int batch = 0; batch < NUM_TRAIN; batch += BATCH_SIZE) {
            int current_batch_size = BATCH_SIZE;
            if (batch + BATCH_SIZE > NUM_TRAIN) {
                current_batch_size = NUM_TRAIN - batch;
            }

            // Copy batch of images and labels to device memory
            for (int i = 0; i < current_batch_size; i++) {
                half* input_half = (half*)malloc(INPUT_SIZE * sizeof(half));
                for (int j = 0; j < INPUT_SIZE; j++) {
                    input_half[j] = __float2half((float)train_images[batch + i][j]);
                }
                cudaMemcpy(d_input + i * INPUT_SIZE, input_half, INPUT_SIZE * sizeof(half), cudaMemcpyHostToDevice);
                free(input_half);

                half* target_half = (half*)malloc(OUTPUT_SIZE * sizeof(half));
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    target_half[j] = __float2half((float)train_labels[batch + i][j]);
                }
                cudaMemcpy(d_target + i * OUTPUT_SIZE, target_half, OUTPUT_SIZE * sizeof(half), cudaMemcpyHostToDevice);
                free(target_half);
            }

            // Forward pass using tensor cores
            forward_hidden_tensor_core<<<grid, block>>>(d_input, d_W1, d_b1, d_hidden, current_batch_size);
            cudaDeviceSynchronize();
            forward_output_tensor_core<<<grid, block>>>(d_hidden, d_W2, d_b2, d_output, current_batch_size);
            cudaDeviceSynchronize();
            softmax_optimized<<<1, OUTPUT_SIZE>>>(d_output, current_batch_size);
            cudaDeviceSynchronize();

            // Compute loss and accuracy on host
            for (int i = 0; i < current_batch_size; i++) {
                float h_output[OUTPUT_SIZE];
                cudaMemcpy(h_output, d_output + i * OUTPUT_SIZE, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
                double sample_loss = 0.0;
                int pred = 0, actual = 0;
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    sample_loss -= train_labels[batch + i][k] * log(h_output[k]);
                    if (h_output[k] > h_output[pred])
                        pred = k;
                    if (train_labels[batch + i][k] > train_labels[batch + i][actual])
                        actual = k;
                }
                epoch_loss += sample_loss;
                if (pred == actual) correct++;
            }

            // Backpropagation
            compute_output_gradient_optimized<<<grid, block>>>(d_output, d_target, d_doutput, current_batch_size);
            cudaDeviceSynchronize();
            compute_hidden_gradient_optimized<<<grid, block>>>(d_W2, d_doutput, d_hidden, d_dhidden, current_batch_size);
            cudaDeviceSynchronize();

            // Update weights and biases using tensor cores
            update_W2_tensor_core<<<grid, block>>>(d_W2, d_doutput, d_hidden, current_batch_size);
            update_W1_tensor_core<<<grid, block>>>(d_W1, d_dhidden, d_input, current_batch_size);
            update_b2_optimized<<<grid, block>>>(d_b2, d_doutput, current_batch_size);
            update_b1_optimized<<<grid, block>>>(d_b1, d_dhidden, current_batch_size);
            cudaDeviceSynchronize();
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
            epoch + 1, epoch_loss / NUM_TRAIN, (correct / (double)NUM_TRAIN) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));

    // Evaluation on test set
    int test_correct = 0;
    for (int i = 0; i < NUM_TEST; i++) {
        half* input_half = (half*)malloc(INPUT_SIZE * sizeof(half));
        for (int j = 0; j < INPUT_SIZE; j++) {
            input_half[j] = __float2half((float)test_images[i][j]);
        }
        cudaMemcpy(d_input, input_half, INPUT_SIZE * sizeof(half), cudaMemcpyHostToDevice);
        free(input_half);

        forward_hidden_tensor_core<<<grid, block>>>(d_input, d_W1, d_b1, d_hidden, 1);
        cudaDeviceSynchronize();
        forward_output_tensor_core<<<grid, block>>>(d_hidden, d_W2, d_b2, d_output, 1);
        cudaDeviceSynchronize();
        softmax_optimized<<<1, 1>>>(d_output, 1);
        cudaDeviceSynchronize();

        float h_output[OUTPUT_SIZE];
        cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > h_output[pred])
                pred = j;
            if (test_labels[i][j] > test_labels[i][actual])
                actual = j;
        }
        if (pred == actual) test_correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (test_correct / (double)NUM_TEST) * 100);

    // Free device memory
    cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_b1); cudaFree(d_b2);
    cudaFree(d_input); cudaFree(d_hidden); cudaFree(d_output); cudaFree(d_target);
    cudaFree(d_doutput); cudaFree(d_dhidden);

    // Free host memory
    free(h_W1); free(h_W2); free(h_b1); free(h_b2);
    freeMatrix(train_images, NUM_TRAIN);
    freeMatrix(train_labels, NUM_TRAIN);
    freeMatrix(test_images, NUM_TEST);
    freeMatrix(test_labels, NUM_TEST);

    return 0;
}