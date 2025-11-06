#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/sort.h>

#define MAX_CITIES 50
#define MAX_CITY_NAME_LENGTH 50
#define MAX_LINE_LENGTH 2048

// City drought risk data
typedef struct {
    char city_name[MAX_CITY_NAME_LENGTH];
    double dri_value;
    double avg_humidity;
    double trend_slope;
    long dry_days;
    long total_records;
    int data_column;
} CityDroughtRisk;

// GPU kernels for parallel computation
__global__ void calculateBasicStats(float* humidity_data, int* record_counts,
                                   double* avg_humidity, long* dry_days,
                                   int num_cities, int total_records) {
    int city_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (city_idx < num_cities && record_counts[city_idx] > 0) {
        double sum = 0.0;
        long dry_count = 0;

        for (int i = 0; i < record_counts[city_idx]; i++) {
            float humidity = humidity_data[city_idx * total_records + i];
            sum += humidity;
            if (humidity < 30.0f) {
                dry_count++;
            }
        }

        avg_humidity[city_idx] = sum / record_counts[city_idx];
        dry_days[city_idx] = dry_count;
    }
}

__global__ void calculateTrends(float* humidity_data, int* record_counts,
                               double* trend_slopes, int num_cities, int total_records) {
    int city_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (city_idx < num_cities && record_counts[city_idx] > 100) {
        int half_count = record_counts[city_idx] / 2;
        double first_half_sum = 0.0, second_half_sum = 0.0;

        for (int i = 0; i < half_count; i++) {
            first_half_sum += humidity_data[city_idx * total_records + i];
        }

        for (int i = half_count; i < record_counts[city_idx]; i++) {
            second_half_sum += humidity_data[city_idx * total_records + i];
        }

        double first_avg = first_half_sum / half_count;
        double second_avg = second_half_sum / (record_counts[city_idx] - half_count);

        trend_slopes[city_idx] = (second_avg - first_avg) / half_count * 100.0;
    } else {
        trend_slopes[city_idx] = 0.0;
    }
}

__global__ void calculateVolatility(float* humidity_data, int* record_counts,
                                   double* volatility, int num_cities, int total_records) {
    int city_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (city_idx < num_cities && record_counts[city_idx] > 1) {
        double total_change = 0.0;

        for (int i = 1; i < record_counts[city_idx]; i++) {
            float curr = humidity_data[city_idx * total_records + i];
            float prev = humidity_data[city_idx * total_records + i - 1];
            total_change += fabs(curr - prev);
        }

        volatility[city_idx] = total_change / (record_counts[city_idx] - 1);
    } else {
        volatility[city_idx] = 0.0;
    }
}

__global__ void calculateDRI(double* avg_humidity, long* dry_days, int* record_counts,
                            double* trend_slopes, double* volatility, double* dri_values,
                            int num_cities) {
    int city_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (city_idx < num_cities && record_counts[city_idx] > 0) {
        // HFI - Humidity Frequency Index
        double HFI = (double)dry_days[city_idx] / record_counts[city_idx];

        // HSI - Humidity Severity Index
        double avg_ratio = avg_humidity[city_idx] / 100.0;
        double HSI = (1.0 - avg_ratio) * 0.6;

        // HTI - Humidity Trend Index
        double HTI = (trend_slopes[city_idx] < 0) ? (-trend_slopes[city_idx] / 2.0) : 0.0;
        if (HTI > 1.0) HTI = 1.0;

        // HVI - Humidity Volatility Index
        double HVI = volatility[city_idx] / 100.0;
        if (HVI > 1.0) HVI = 1.0;

        // Calculate final DRI
        dri_values[city_idx] = 0.35 * HFI + 0.30 * HSI + 0.20 * HTI + 0.15 * HVI;
        if (dri_values[city_idx] > 1.0) dri_values[city_idx] = 1.0;
    }
}

// Comparison function for sorting
int compareDRI(const void* a, const void* b) {
    CityDroughtRisk* cityA = (CityDroughtRisk*)a;
    CityDroughtRisk* cityB = (CityDroughtRisk*)b;
    if (cityB->dri_value > cityA->dri_value) return 1;
    if (cityB->dri_value < cityA->dri_value) return -1;
    return 0;
}

// Global variables
CityDroughtRisk cities[MAX_CITIES];
int num_cities = 0;
int total_records = 0;

// Parse city names from header line
int parseCityHeaders(const char* header_line) {
    char* token;
    char line_copy[MAX_LINE_LENGTH];
    int column = 0;

    strcpy(line_copy, header_line);

    // Skip datetime column
    token = strtok(line_copy, ",");
    column++;

    // Parse city names
    while ((token = strtok(NULL, ",")) != NULL && num_cities < MAX_CITIES) {
        if (strlen(token) > 0) {
            strcpy(cities[num_cities].city_name, token);
            cities[num_cities].data_column = column;
            cities[num_cities].dri_value = 0.0;
            cities[num_cities].avg_humidity = 0.0;
            cities[num_cities].trend_slope = 0.0;
            cities[num_cities].dry_days = 0;
            cities[num_cities].total_records = 0;
            num_cities++;
        }
        column++;
    }

    return num_cities;
}

// Load multi-city data from CSV
void loadMultiCityData(const char* filepath) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filepath);
        return;
    }

    char line[MAX_LINE_LENGTH];

    // Read header line
    if (fgets(line, sizeof(line), file)) {
        parseCityHeaders(line);
    }

    fclose(file);
}

// Process all data with GPU acceleration
void processAllDataGPU(const char* filepath) {
    FILE* file = fopen(filepath, "r");
    if (!file) return;

    char line[MAX_LINE_LENGTH];

    // Skip header
    fgets(line, sizeof(line), file);

    // Allocate memory for humidity data
    float** humidity_data = (float**)malloc((num_cities + 5) * sizeof(float*));
    for (int i = 0; i < num_cities + 5; i++) {
        humidity_data[i] = (float*)malloc(100000 * sizeof(float));
    }

    int record_count = 0;

    // Read data line by line
    while (fgets(line, sizeof(line), file) && record_count < 100000) {
        char* token = strtok(line, ",");
        int column = 0;

        // Skip datetime column
        column++;

        while ((token = strtok(NULL, ",")) != NULL && column < num_cities + 5) {
            float humidity = atof(token);
            if (humidity > 0.0f && humidity <= 100.0f) {
                humidity_data[column][record_count] = humidity;
                if (column - 1 < num_cities) {
                    cities[column - 1].total_records++;
                }
            }
            column++;
        }
        record_count++;
    }

    total_records = record_count;
    fclose(file);

    // Prepare data for GPU
    int* device_record_counts;
    double* device_avg_humidity;
    long* device_dry_days;
    double* device_trend_slopes;
    double* device_volatility;
    double* device_dri_values;
    float* device_humidity_data;

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc(&device_record_counts, num_cities * sizeof(int));
    if (err != cudaSuccess) printf("CUDA Error 1: %s\n", cudaGetErrorString(err));

    err = cudaMalloc(&device_avg_humidity, num_cities * sizeof(double));
    if (err != cudaSuccess) printf("CUDA Error 2: %s\n", cudaGetErrorString(err));

    err = cudaMalloc(&device_dry_days, num_cities * sizeof(long));
    if (err != cudaSuccess) printf("CUDA Error 3: %s\n", cudaGetErrorString(err));

    err = cudaMalloc(&device_trend_slopes, num_cities * sizeof(double));
    if (err != cudaSuccess) printf("CUDA Error 4: %s\n", cudaGetErrorString(err));

    err = cudaMalloc(&device_volatility, num_cities * sizeof(double));
    if (err != cudaSuccess) printf("CUDA Error 5: %s\n", cudaGetErrorString(err));

    err = cudaMalloc(&device_dri_values, num_cities * sizeof(double));
    if (err != cudaSuccess) printf("CUDA Error 6: %s\n", cudaGetErrorString(err));

    err = cudaMalloc(&device_humidity_data, num_cities * total_records * sizeof(float));
    if (err != cudaSuccess) printf("CUDA Error 7: %s\n", cudaGetErrorString(err));

    // Copy record counts to device
    int* host_record_counts = (int*)malloc(num_cities * sizeof(int));
    for (int i = 0; i < num_cities; i++) {
        host_record_counts[i] = cities[i].total_records;
    }
    cudaMemcpy(device_record_counts, host_record_counts, num_cities * sizeof(int), cudaMemcpyHostToDevice);

    // Copy humidity data to device (flatten 2D array)
    float* flattened_data = (float*)malloc(num_cities * total_records * sizeof(float));
    for (int city = 0; city < num_cities; city++) {
        for (int record = 0; record < cities[city].total_records; record++) {
            flattened_data[city * total_records + record] = humidity_data[city][record];
        }
    }
    cudaMemcpy(device_humidity_data, flattened_data, num_cities * total_records * sizeof(float), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (num_cities + threads_per_block - 1) / threads_per_block;

    // Launch kernels
    calculateBasicStats<<<blocks_per_grid, threads_per_block>>>(
        device_humidity_data, device_record_counts, device_avg_humidity,
        device_dry_days, num_cities, total_records);

    calculateTrends<<<blocks_per_grid, threads_per_block>>>(
        device_humidity_data, device_record_counts, device_trend_slopes,
        num_cities, total_records);

    calculateVolatility<<<blocks_per_grid, threads_per_block>>>(
        device_humidity_data, device_record_counts, device_volatility,
        num_cities, total_records);

    calculateDRI<<<blocks_per_grid, threads_per_block>>>(
        device_avg_humidity, device_dry_days, device_record_counts,
        device_trend_slopes, device_volatility, device_dri_values, num_cities);

    cudaDeviceSynchronize();

    // Copy results back to host
    double* host_avg_humidity = (double*)malloc(num_cities * sizeof(double));
    long* host_dry_days = (long*)malloc(num_cities * sizeof(long));
    double* host_trend_slopes = (double*)malloc(num_cities * sizeof(double));
    double* host_volatility = (double*)malloc(num_cities * sizeof(double));
    double* host_dri_values = (double*)malloc(num_cities * sizeof(double));

    cudaMemcpy(host_avg_humidity, device_avg_humidity, num_cities * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dry_days, device_dry_days, num_cities * sizeof(long), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_trend_slopes, device_trend_slopes, num_cities * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_volatility, device_volatility, num_cities * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dri_values, device_dri_values, num_cities * sizeof(double), cudaMemcpyDeviceToHost);

    // Update cities data
    for (int i = 0; i < num_cities; i++) {
        cities[i].avg_humidity = host_avg_humidity[i];
        cities[i].dry_days = host_dry_days[i];
        cities[i].trend_slope = host_trend_slopes[i];
        cities[i].dri_value = host_dri_values[i];
    }

    // Sort cities by DRI using CPU
    qsort(cities, num_cities, sizeof(CityDroughtRisk), compareDRI);

    // Free memory
    cudaFree(device_record_counts);
    cudaFree(device_avg_humidity);
    cudaFree(device_dry_days);
    cudaFree(device_trend_slopes);
    cudaFree(device_volatility);
    cudaFree(device_dri_values);
    cudaFree(device_humidity_data);

    for (int i = 0; i < num_cities + 5; i++) {
        free(humidity_data[i]);
    }
    free(humidity_data);
    free(host_record_counts);
    free(flattened_data);
    free(host_avg_humidity);
    free(host_dry_days);
    free(host_trend_slopes);
    free(host_volatility);
    free(host_dri_values);
}

// Print top 3 cities with highest drought risk
void printTop3Cities(double process_time) {
    printf("Dataset: %d records\n", total_records);
    printf("Top 3 Drought Risk Cities:\n");

    for (int i = 0; i < 3 && i < num_cities; i++) {
        printf("%d. %s (DRI: %.3f)\n", i + 1, cities[i].city_name, cities[i].dri_value);
    }

    printf("Time: %.3f seconds\n", process_time);
}

// Main program
int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <humidity_data.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    clock_t start_time = clock();

    loadMultiCityData(argv[1]);
    processAllDataGPU(argv[1]);

    clock_t end_time = clock();
    double process_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printTop3Cities(process_time);

    return EXIT_SUCCESS;
}