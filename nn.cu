#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 10  // Increased from 6 to 10
#define BATCH_SIZE 128  // Increased from 64 to 128 for better convergence
#define BLOCK_SIZE 256

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

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// CUDA kernel for matrix-vector multiplication
__global__ void matrixVectorMulKernel(float* d_matrix, float* d_input, float* d_output, 
                                     float* d_bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < rows) {
        float sum = d_bias[idx];
        for (int j = 0; j < cols; j++) {
            sum += d_matrix[idx * cols + j] * d_input[j];
        }
        d_output[idx] = sum;
    }
}

// CUDA kernel for ReLU activation
__global__ void reluKernel(float* d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        d_data[idx] = (d_data[idx] > 0) ? d_data[idx] : 0;
    }
}

// CUDA kernel for softmax
__global__ void softmaxKernel(float* d_data, int size) {
    // First find max for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < size; i++) {
        if (d_data[i] > max_val) {
            max_val = d_data[i];
        }
    }
    
    // Calculate exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        d_data[i] = expf(d_data[i] - max_val);
        sum += d_data[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        d_data[i] /= sum;
    }
}

// CUDA kernel for computing output gradients
__global__ void outputGradientKernel(float* d_output, float* d_target, float* d_gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        d_gradient[idx] = d_output[idx] - d_target[idx];
    }
}

// CUDA kernel for computing hidden gradients
__global__ void hiddenGradientKernel(float* d_W2, float* d_output_gradient, 
                                    float* d_hidden, float* d_hidden_gradient, 
                                    int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < hidden_size) {
        float sum = 0.0f;
        for (int j = 0; j < output_size; j++) {
            sum += d_W2[j * hidden_size + idx] * d_output_gradient[j];
        }
        // ReLU derivative
        d_hidden_gradient[idx] = sum * (d_hidden[idx] > 0 ? 1.0f : 0.0f);
    }
}

// CUDA kernel for updating weights
__global__ void updateWeightsKernel(float* d_weights, float* d_gradient, float* d_inputs,
                                  int rows, int cols, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / cols;
    int col = idx % cols;
    
    if (row < rows && col < cols) {
        d_weights[row * cols + col] -= learning_rate * d_gradient[row] * d_inputs[col];
    }
}

// CUDA kernel for updating biases
__global__ void updateBiasKernel(float* d_bias, float* d_gradient, 
                               int size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        d_bias[idx] -= learning_rate * d_gradient[idx];
    }
}

// Simple neural network structure
typedef struct {
    // Host matrices (flattened for easier management)
    float* W1;
    float* W2;
    float* b1;
    float* b2;
    
    // Device matrices
    float* d_W1;
    float* d_W2;
    float* d_b1;
    float* d_b2;
    
    // Temporary storage
    float* d_input;
    float* d_hidden;
    float* d_output;
    float* d_target;
    float* d_hidden_gradient;
    float* d_output_gradient;
    
    // Batch storage
    float* d_batch_input;
    float* d_batch_hidden;
    float* d_batch_output;
    float* d_batch_target;
    float* d_batch_hidden_gradient;
    float* d_batch_output_gradient;
} NeuralNetwork;

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Allocate host memory (flattened for simplicity)
    net->W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    net->W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    net->b1 = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    net->b2 = (float*)calloc(OUTPUT_SIZE, sizeof(float));

    // Initialize weights with Xavier/Glorot initialization
    srand(time(NULL));
    float w1_range = sqrtf(6.0f / (INPUT_SIZE + HIDDEN_SIZE));
    float w2_range = sqrtf(6.0f / (HIDDEN_SIZE + OUTPUT_SIZE));
    
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        net->W1[i] = ((float)rand() / RAND_MAX) * 2 * w1_range - w1_range;
    }

    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        net->W2[i] = ((float)rand() / RAND_MAX) * 2 * w2_range - w2_range;
    }
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_b1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_b2, OUTPUT_SIZE * sizeof(float)));
    
    // Individual sample memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_input, INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_hidden, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_output, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_target, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_hidden_gradient, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_output_gradient, OUTPUT_SIZE * sizeof(float)));
    
    // Batch memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_input, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_hidden_gradient, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_output_gradient, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    // Copy initial weights to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    return net;
}

// Forward pass (for evaluation)
void forward(NeuralNetwork* net, float* input) {
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Calculate hidden layer: hidden = ReLU(W1 * input + b1)
    dim3 hiddenBlocks((HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixVectorMulKernel<<<hiddenBlocks, BLOCK_SIZE>>>(
        net->d_W1, net->d_input, net->d_hidden, net->d_b1, HIDDEN_SIZE, INPUT_SIZE);
    
    reluKernel<<<hiddenBlocks, BLOCK_SIZE>>>(net->d_hidden, HIDDEN_SIZE);
    
    // Calculate output layer: output = softmax(W2 * hidden + b2)
    dim3 outputBlocks((OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixVectorMulKernel<<<outputBlocks, BLOCK_SIZE>>>(
        net->d_W2, net->d_hidden, net->d_output, net->d_b2, OUTPUT_SIZE, HIDDEN_SIZE);
    
    softmaxKernel<<<1, 1>>>(net->d_output, OUTPUT_SIZE);
}

// Train the neural network using mini-batch SGD
void train(NeuralNetwork* net, float** images, float** labels, int numImages) {
    clock_t total_start = clock();
    
    // Prepare flattened batch arrays
    float* batch_input = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    float* batch_target = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    float* output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        
        // Random shuffle indices for stochastic sampling
        int* indices = (int*)malloc(numImages * sizeof(int));
        for (int i = 0; i < numImages; i++) {
            indices[i] = i;
        }
        for (int i = 0; i < numImages; i++) {
            int j = rand() % numImages;
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        // Process data in batches
        for (int b = 0; b < numImages; b += BATCH_SIZE) {
            int batchSize = (b + BATCH_SIZE <= numImages) ? BATCH_SIZE : (numImages - b);
            
            // Prepare batch data
            for (int i = 0; i < batchSize; i++) {
                int idx = indices[b + i];
                for (int j = 0; j < INPUT_SIZE; j++) {
                    batch_input[i * INPUT_SIZE + j] = images[idx][j];
                }
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    batch_target[i * OUTPUT_SIZE + j] = labels[idx][j];
                }
            }
            
            // Copy batch data to device
            CHECK_CUDA_ERROR(cudaMemcpy(net->d_batch_input, batch_input, 
                              batchSize * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(net->d_batch_target, batch_target, 
                              batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            
            // Zeroing out gradient accumulators (could use cudaMemset for this)
            float* zero_hidden = (float*)calloc(HIDDEN_SIZE, sizeof(float));
            float* zero_output = (float*)calloc(OUTPUT_SIZE, sizeof(float));
            CHECK_CUDA_ERROR(cudaMemcpy(net->d_hidden_gradient, zero_hidden, 
                              HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(net->d_output_gradient, zero_output, 
                              OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            free(zero_hidden);
            free(zero_output);
            
            // Process each sample in batch
            for (int i = 0; i < batchSize; i++) {
                // Extract individual sample from batch
                CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, &batch_input[i * INPUT_SIZE], 
                                  INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA_ERROR(cudaMemcpy(net->d_target, &batch_target[i * OUTPUT_SIZE], 
                                  OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
                
                // Forward pass
                forward(net, &batch_input[i * INPUT_SIZE]);
                
                // Get output for loss calculation
                CHECK_CUDA_ERROR(cudaMemcpy(output, net->d_output, 
                                  OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
                
                // Calculate loss (cross-entropy)
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    if (batch_target[i * OUTPUT_SIZE + k] > 0.5) { // One-hot encoded
                        loss -= logf(output[k] > 1e-10f ? output[k] : 1e-10f);
                    }
                }
                
                // Check accuracy
                int pred = 0;
                for (int j = 1; j < OUTPUT_SIZE; j++) {
                    if (output[j] > output[pred]) pred = j;
                }
                int actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (batch_target[i * OUTPUT_SIZE + j] > 0.5) {
                        actual = j;
                        break;
                    }
                }
                if (pred == actual) correct++;
                
                // Calculate gradients
                outputGradientKernel<<<(OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                    net->d_output, net->d_target, net->d_output_gradient, OUTPUT_SIZE);
                
                hiddenGradientKernel<<<(HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                    net->d_W2, net->d_output_gradient, 
                    net->d_hidden, net->d_hidden_gradient, 
                    HIDDEN_SIZE, OUTPUT_SIZE);
                
                // Update weights and biases
                updateWeightsKernel<<<(OUTPUT_SIZE * HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                    net->d_W2, net->d_output_gradient, net->d_hidden, 
                    OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE / batchSize);
                
                updateWeightsKernel<<<(HIDDEN_SIZE * INPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                    net->d_W1, net->d_hidden_gradient, net->d_input, 
                    HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE / batchSize);
                
                updateBiasKernel<<<(OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                    net->d_b2, net->d_output_gradient, OUTPUT_SIZE, LEARNING_RATE / batchSize);
                
                updateBiasKernel<<<(HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                    net->d_b1, net->d_hidden_gradient, HIDDEN_SIZE, LEARNING_RATE / batchSize);
            }
        }
        
        // Copy model parameters back to host (only needed occasionally)
        if (epoch % 5 == 0 || epoch == EPOCHS - 1) {
            CHECK_CUDA_ERROR(cudaMemcpy(net->W1, net->d_W1, 
                            HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(net->W2, net->d_W2, 
                            OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(net->b1, net->d_b1, 
                            HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(net->b2, net->d_b2, 
                            OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        }
        
        free(indices);
        
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    
    free(batch_input);
    free(batch_target);
    free(output);
    
    printf("Total training time: %.3fs\n", get_time(total_start));
}

// Evaluate the neural network
void evaluate(NeuralNetwork* net, float** images, float** labels, int numImages) {
    clock_t eval_start = clock();
    int correct = 0;
    float* output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    for (int i = 0; i < numImages; i++) {
        // Forward pass
        forward(net, images[i]);
        
        // Get predictions
        CHECK_CUDA_ERROR(cudaMemcpy(output, net->d_output, 
                          OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Find predicted class
        int pred = 0;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
        }
        
        // Find actual class
        int actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (labels[i][j] > 0.5) {
                actual = j;
                break;
            }
        }
        
        if (pred == actual) correct++;
    }
    
    free(output);
    
    printf("Test Accuracy: %.2f%% - Evaluation Time: %.3fs\n", 
           (correct / (double)numImages) * 100, get_time(eval_start));
}

// Free neural network memory
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
    cudaFree(net->d_batch_input);
    cudaFree(net->d_batch_hidden);
    cudaFree(net->d_batch_output);
    cudaFree(net->d_batch_target);
    cudaFree(net->d_batch_hidden_gradient);
    cudaFree(net->d_batch_output_gradient);
    
    // Free host memory
    free(net->W1);
    free(net->W2);
    free(net->b1);
    free(net->b2);
    free(net);
}

// Load MNIST images
float** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    
    // Skip header
    fseek(file, 16, SEEK_SET);
    
    // Allocate memory
    float** images = (float**)malloc(numImages * sizeof(float*));
    for (int i = 0; i < numImages; i++) {
        images[i] = (float*)malloc(INPUT_SIZE * sizeof(float));
    }
    
    // Read pixel data
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0f;  // Normalize to [0,1]
        }
    }
    
    fclose(file);
    return images;
}

// Load MNIST labels
float** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    
    // Skip header
    fseek(file, 8, SEEK_SET);
    
    // Allocate memory
    float** labels = (float**)malloc(numLabels * sizeof(float*));
    for (int i = 0; i < numLabels; i++) {
        labels[i] = (float*)calloc(OUTPUT_SIZE, sizeof(float));
    }
    
    // Read label data
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        labels[i][label] = 1.0f;  // One-hot encoding
    }
    
    fclose(file);
    return labels;
}

int main() {
    printf("MNIST Neural Network with CUDA Implementation\n\n");
    
    // Print CUDA device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("CUDA Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Shared memory per block: %ld KB\n\n", prop.sharedMemPerBlock / 1024);

    // Load MNIST dataset
    float** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    float** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    float** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    float** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);

    // Create and train neural network
    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    // Free memory
    freeNetwork(net);
    for (int i = 0; i < 60000; i++) {
        free(train_images[i]);
        free(train_labels[i]);
    }
    for (int i = 0; i < 10000; i++) {
        free(test_images[i]);
        free(test_labels[i]);
    }
    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);
    
    // Reset CUDA device
    cudaDeviceReset();
    
    return 0;
}