#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// Network parameters
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.1
#define EPOCHS 8
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define BATCH_SIZE 128

// Tensor Core optimized forward pass for hidden layer
__global__ void forward_hidden_tensorcore(const double* d_input, const double* d_W1, const double* d_b1, double* d_hidden, int batch_size) {
    // Using mixed precision with Tensor Cores
    using namespace nvcuda;

    // Declare fragments for WMMA
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize the accumulator to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    // Get batch and hidden neuron indices
    int batch_idx = blockIdx.y;
    int hid_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || hid_idx >= HIDDEN_SIZE) return;

   // Convert bias to float
   float bias = (float)d_b1[hid_idx];

    // Shared memory for input tiles (pad to multiple of 16)
   __shared__ __half input_tile[16][16];

   // Perform the matrix multiplication using Tensor Cores
   for (int tile = 0; tile < (INPUT_SIZE + 15) / 16; tile++) {
   int input_offset = tile * 16;

   // Load weights tile (convert from FP64 to FP16)
   __half weights_tile[16];
   for (int i = 0; i < 16 && (input_offset + i) < INPUT_SIZE; i++) {
       weights_tile[i] = __double2half(d_W1[hid_idx * INPUT_SIZE + input_offset + i]);
    }
    wmma::load_matrix_sync(a_frag, weights_tile, INPUT_SIZE);

    // Load input tile (convert from FP64 to FP16)
    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        int input_idx = input_offset + i;
        if (input_idx < INPUT_SIZE) {
        input_tile[i][threadIdx.x] = __double2half(d_input[batch_idx * INPUT_SIZE + input_idx]);
    }
    else {
        input_tile[i][threadIdx.x] = __float2half(0.0f); // Zero padding
    }
    }
    __syncthreads();

    wmma::load_matrix_sync(b_frag, &input_tile[0][0], 16);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    __syncthreads();
    }

        // Store the result (convert back to FP64)
        float result = 0.0f;
        for (int i = 0; i < acc_frag.num_elements; i++) {
            result += acc_frag.x[i];
        }
        result += bias;
        d_hidden[batch_idx * HIDDEN_SIZE + hid_idx] = (result > 0) ? (double)result : 0.0;
    }

// Timer helper
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Optimized atomicAdd for double
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

// Optimized forward_hidden kernel
__global__ void forward_hidden_optimized(const double* d_input, const double* d_W1, const double* d_b1, double* d_hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < HIDDEN_SIZE) {
        double sum = d_b1[idx];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += d_W1[idx * INPUT_SIZE + j] * d_input[j];
        }
        d_hidden[idx] = (sum > 0) ? sum : 0;
    }
}

// Optimized forward_output kernel
__global__ void forward_output_optimized(const double* d_hidden, const double* d_W2, const double* d_b2, double* d_output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < OUTPUT_SIZE) {
        double sum = d_b2[idx];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += d_W2[idx * HIDDEN_SIZE + j] * d_hidden[j];
        }
        d_output[idx] = sum;
    }
}

// Optimized softmax kernel
__global__ void softmax_optimized(double* d_output) {
    __shared__ double sum_shared[OUTPUT_SIZE];
    int idx = threadIdx.x;
    if (idx == 0) {
        sum_shared[0] = 0;
    }
    __syncthreads();

    if (idx < OUTPUT_SIZE) {
        d_output[idx] = exp(d_output[idx]);
        atomicAddDouble(&sum_shared[0], d_output[idx]);
    }
    __syncthreads();

    if (idx < OUTPUT_SIZE) {
        d_output[idx] /= sum_shared[0];
    }
}

// Batch processing optimized forward_hidden kernel
__global__ void forward_hidden_batch(const double* d_input, const double* d_W1, const double* d_b1, double* d_hidden, int batch_size) {
    int batch_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < HIDDEN_SIZE && batch_idx < batch_size) {
        double sum = d_b1[idx];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += d_W1[idx * INPUT_SIZE + j] * d_input[batch_idx * INPUT_SIZE + j];
        }
        d_hidden[batch_idx * HIDDEN_SIZE + idx] = (sum > 0) ? sum : 0;
    }
}

// Batch processing optimized forward_output kernel
__global__ void forward_output_batch(const double* d_hidden, const double* d_W2, const double* d_b2, double* d_output, int batch_size) {
    int batch_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < OUTPUT_SIZE && batch_idx < batch_size) {
        double sum = d_b2[idx];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += d_W2[idx * HIDDEN_SIZE + j] * d_hidden[batch_idx * HIDDEN_SIZE + j];
        }
        d_output[batch_idx * OUTPUT_SIZE + idx] = sum;
    }
}

// Batch processing optimized softmax kernel
__global__ void softmax_batch(double* d_output, int batch_size) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    __shared__ double sum_shared;
    if (threadIdx.x == 0) sum_shared = 0;
    __syncthreads();

    int idx = threadIdx.x;
    if (idx < OUTPUT_SIZE) {
        d_output[batch_idx * OUTPUT_SIZE + idx] = exp(d_output[batch_idx * OUTPUT_SIZE + idx]);
        atomicAddDouble(&sum_shared, d_output[batch_idx * OUTPUT_SIZE + idx]);
    }
    __syncthreads();

    if (idx < OUTPUT_SIZE) {
        d_output[batch_idx * OUTPUT_SIZE + idx] /= sum_shared;
    }
}

// Batch processing optimized compute_output_gradient kernel
__global__ void compute_output_gradient_batch(const double* d_output, const double* d_target, double* d_doutput, int batch_size) {
    int batch_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < OUTPUT_SIZE && batch_idx < batch_size) {
        d_doutput[batch_idx * OUTPUT_SIZE + idx] = d_output[batch_idx * OUTPUT_SIZE + idx] - d_target[batch_idx * OUTPUT_SIZE + idx];
    }
}

// Batch processing optimized compute_hidden_gradient kernel
__global__ void compute_hidden_gradient_batch(const double* d_W2, const double* d_doutput, const double* d_hidden, double* d_dhidden, int batch_size) {
    int batch_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < HIDDEN_SIZE && batch_idx < batch_size) {
        double sum = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            sum += d_W2[j * HIDDEN_SIZE + idx] * d_doutput[batch_idx * OUTPUT_SIZE + j];
        }
        d_dhidden[batch_idx * HIDDEN_SIZE + idx] = (d_hidden[batch_idx * HIDDEN_SIZE + idx] > 0) ? sum : 0;
    }
}

// Batch processing optimized update_W2 kernel
__global__ void update_W2_batch(double* d_W2, const double* d_doutput, const double* d_hidden, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = OUTPUT_SIZE * HIDDEN_SIZE;
    if (idx < total) {
        int out_idx = idx / HIDDEN_SIZE;
        int hid_idx = idx % HIDDEN_SIZE;
        double grad = 0.0;
        for (int b = 0; b < batch_size; b++) {
            grad += d_doutput[b * OUTPUT_SIZE + out_idx] * d_hidden[b * HIDDEN_SIZE + hid_idx];
        }
        d_W2[idx] -= LEARNING_RATE * grad / batch_size;
    }
}

// Batch processing optimized update_W1 kernel
__global__ void update_W1_batch(double* d_W1, const double* d_dhidden, const double* d_input, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = HIDDEN_SIZE * INPUT_SIZE;
    if (idx < total) {
        int hid_idx = idx / INPUT_SIZE;
        int in_idx = idx % INPUT_SIZE;
        double grad = 0.0;
        for (int b = 0; b < batch_size; b++) {
            grad += d_dhidden[b * HIDDEN_SIZE + hid_idx] * d_input[b * INPUT_SIZE + in_idx];
        }
        d_W1[idx] -= LEARNING_RATE * grad / batch_size;
    }
}

// Batch processing optimized update_b2 kernel
__global__ void update_b2_batch(double* d_b2, const double* d_doutput, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < OUTPUT_SIZE) {
        double grad = 0.0;
        for (int b = 0; b < batch_size; b++) {
            grad += d_doutput[b * OUTPUT_SIZE + idx];
        }
        d_b2[idx] -= LEARNING_RATE * grad / batch_size;
    }
}

// Batch processing optimized update_b1 kernel
__global__ void update_b1_batch(double* d_b1, const double* d_dhidden, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < HIDDEN_SIZE) {
        double grad = 0.0;
        for (int b = 0; b < batch_size; b++) {
            grad += d_dhidden[b * HIDDEN_SIZE + idx];
        }
        d_b1[idx] -= LEARNING_RATE * grad / batch_size;
    }
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

int main() {
    printf("Further Optimized GPU Implementation for MNIST Neural Network\n\n");
    clock_t total_start = clock();

    // Load data on host
    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", NUM_TRAIN);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", NUM_TRAIN);
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", NUM_TEST);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", NUM_TEST);

    // Allocate device memory for network parameters
    double *d_W1, *d_W2, *d_b1, *d_b2;
    cudaMalloc((void**)&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&d_b1, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&d_b2, OUTPUT_SIZE * sizeof(double));

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

    // Copy network parameters to device
    cudaMemcpy(d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate device memory for per-batch data and intermediates
    double *d_input, *d_hidden, *d_output, *d_target;
    double *d_doutput, *d_dhidden;
    cudaMalloc((void**)&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&d_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&d_doutput, BATCH_SIZE * OUTPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&d_dhidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(double));

    // Optimized kernel launch configuration for batch processing
    dim3 hiddenBlock(256, 1, 1);
    dim3 outputBlock(256, 1, 1);
    int hiddenGrid = (HIDDEN_SIZE + hiddenBlock.x - 1) / hiddenBlock.x;
    int outputGrid = (OUTPUT_SIZE + outputBlock.x - 1) / outputBlock.x;

    // Training loop with batch processing
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
                cudaMemcpy(d_input + i * INPUT_SIZE, train_images[batch + i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(d_target + i * OUTPUT_SIZE, train_labels[batch + i], OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
            }

            // Forward pass
            //forward_hidden_batch<<<dim3(hiddenGrid, current_batch_size), hiddenBlock>>>(d_input, d_W1, d_b1, d_hidden, current_batch_size);
            //forward_hidden_tensorcore<<<dim3((HIDDEN_SIZE + 255) / 256, BATCH_SIZE), 256>>>(d_input, d_W1, d_b1, d_hidden, current_batch_size);
            
            dim3 blockDim(32); // Optimal for WMMA
            dim3 gridDim((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, BATCH_SIZE);
            forward_hidden_tensorcore<<<gridDim, blockDim>>>(d_input, d_W1, d_b1, d_hidden, current_batch_size);
            
            cudaDeviceSynchronize();
            forward_output_batch<<<dim3(outputGrid, current_batch_size), outputBlock>>>(d_hidden, d_W2, d_b2, d_output, current_batch_size);
            cudaDeviceSynchronize();
            softmax_batch<<<dim3(1, current_batch_size), OUTPUT_SIZE>>>(d_output, current_batch_size);
            cudaDeviceSynchronize();

            // Compute loss and accuracy on host
            for (int i = 0; i < current_batch_size; i++) {
                double h_output[OUTPUT_SIZE];
                cudaMemcpy(h_output, d_output + i * OUTPUT_SIZE, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
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
            compute_output_gradient_batch<<<dim3(outputGrid, current_batch_size), outputBlock>>>(d_output, d_target, d_doutput, current_batch_size);
            cudaDeviceSynchronize();
            compute_hidden_gradient_batch<<<dim3(hiddenGrid, current_batch_size), hiddenBlock>>>(d_W2, d_doutput, d_hidden, d_dhidden, current_batch_size);
            cudaDeviceSynchronize();

            // Update weights and biases
            int total_W2 = OUTPUT_SIZE * HIDDEN_SIZE;
            update_W2_batch<<<(total_W2 + 255) / 256, 256>>>(d_W2, d_doutput, d_hidden, current_batch_size);
            int total_W1 = HIDDEN_SIZE * INPUT_SIZE;
            update_W1_batch<<<(total_W1 + 255) / 256, 256>>>(d_W1, d_dhidden, d_input, current_batch_size);
            update_b2_batch<<<outputGrid, outputBlock>>>(d_b2, d_doutput, current_batch_size);
            update_b1_batch<<<hiddenGrid, hiddenBlock>>>(d_b1, d_dhidden, current_batch_size);
            cudaDeviceSynchronize();
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
            epoch + 1, epoch_loss / NUM_TRAIN, (correct / (double)NUM_TRAIN) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));

    // Evaluation on test set
    int test_correct = 0;
    for (int i = 0; i < NUM_TEST; i++) {
        cudaMemcpy(d_input, test_images[i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        forward_hidden_optimized<<<hiddenGrid, hiddenBlock>>>(d_input, d_W1, d_b1, d_hidden);
        cudaDeviceSynchronize();
        forward_output_optimized<<<outputGrid, outputBlock>>>(d_hidden, d_W2, d_b2, d_output);
        cudaDeviceSynchronize();
        softmax_optimized<<<1, OUTPUT_SIZE>>>(d_output);
        cudaDeviceSynchronize();
        double h_output[OUTPUT_SIZE];
        cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
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