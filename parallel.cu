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
#include <thrust/scan.h>

typedef struct {
    int year;
    int month;
    float humidity;
} HumidityRecord;

typedef struct {
    float mean_humidity;
    float min_humidity;
    float max_humidity;
    float humidity_trend;
    long record_count;
    int best_year;
    int worst_year;
    float humidity_variance;
} HumidityStats;

// CUDA kernel to extract humidity data
__global__ void extractHumidityKernel(const HumidityRecord* records, float* humidity,
                                     int* years, long count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        humidity[idx] = records[idx].humidity;
        years[idx] = records[idx].year;
    }
}

HumidityRecord* readHumidityCSV(const char* filename, long* count) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open %s\n", filename);
        return NULL;
    }

    char line[1024];
    long lines = 0;
    while (fgets(line, sizeof(line), file)) lines++;
    rewind(file);

    // Skip header line
    if (fgets(line, sizeof(line), file) == NULL) {
        printf("Error: Empty file or header read failed\n");
        fclose(file);
        return NULL;
    }
    lines--;

    if (lines <= 0) {
        printf("Error: No data records found in file\n");
        fclose(file);
        return NULL;
    }

    HumidityRecord* records = malloc(lines * sizeof(HumidityRecord));
    if (!records) {
        printf("Error: Memory allocation failed for %ld records\n", lines);
        fclose(file);
        return NULL;
    }

    long count_read = 0;
    long line_number = 2; // Start after header

    while (fgets(line, sizeof(line), file) && count_read < lines) {
        HumidityRecord r;
        if (sscanf(line, "%d,%d,%f", &r.year, &r.month, &r.humidity) == 3) {
            // Basic validation
            if (r.year < 1900 || r.year > 2100 || r.month < 1 || r.month > 12) {
                printf("Warning: Invalid date %d/%d at line %ld, skipping\n", r.year, r.month, line_number);
                line_number++;
                continue;
            }
            records[count_read++] = r;
        } else {
            printf("Warning: Malformed line %ld, skipping: %s", line_number, line);
        }
        line_number++;
    }

    fclose(file);

    if (count_read == 0) {
        printf("Error: No valid records found in file\n");
        free(records);
        return NULL;
    }

    *count = count_read;
    printf("Loaded %ld humidity records (out of %ld lines processed)\n", count_read, lines);
    return records;
}

HumidityStats analyzeParallel(HumidityRecord* records, long count) {
    HumidityStats stats = {0};
    stats.record_count = count;

    if (count == 0) return stats;

    clock_t start = clock();

    // Allocate device memory with error checking
    HumidityRecord* d_records = NULL;
    float* d_humidity = NULL;
    int* d_years = NULL;

    cudaError_t err;
    err = cudaMalloc(&d_records, count * sizeof(HumidityRecord));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate device memory for records: %s\n", cudaGetErrorString(err));
        return stats;
    }

    err = cudaMalloc(&d_humidity, count * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate device memory for humidity: %s\n", cudaGetErrorString(err));
        cudaFree(d_records);
        return stats;
    }

    err = cudaMalloc(&d_years, count * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate device memory for years: %s\n", cudaGetErrorString(err));
        cudaFree(d_records);
        cudaFree(d_humidity);
        return stats;
    }

    // Copy to device with error checking
    err = cudaMemcpy(d_records, records, count * sizeof(HumidityRecord), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy data to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_records);
        cudaFree(d_humidity);
        cudaFree(d_years);
        return stats;
    }

    // Setup kernel
    int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;

    // Launch kernel to extract data with error checking
    extractHumidityKernel<<<gridSize, blockSize>>>(d_records, d_humidity, d_years, count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_records);
        cudaFree(d_humidity);
        cudaFree(d_years);
        return stats;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: Kernel synchronization failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_records);
        cudaFree(d_humidity);
        cudaFree(d_years);
        return stats;
    }

    // Use Thrust for parallel operations
    thrust::device_ptr<float> d_humidity_ptr(d_humidity);
    thrust::device_ptr<int> d_years_ptr(d_years);

    // THRUST REDUCE: Parallel sum reduction
    float humidity_sum = thrust::reduce(d_humidity_ptr, d_humidity_ptr + count, 0.0f);

    // THRUST EXTREMA: Find min/max
    float min_humidity = *thrust::min_element(d_humidity_ptr, d_humidity_ptr + count);
    float max_humidity = *thrust::max_element(d_humidity_ptr, d_humidity_ptr + count);

    // Calculate mean
    stats.mean_humidity = humidity_sum / count;
    stats.min_humidity = min_humidity;
    stats.max_humidity = max_humidity;

    // Copy back data for trend analysis with error checking
    float* h_humidity = (float*)malloc(count * sizeof(float));
    int* h_years = (int*)malloc(count * sizeof(int));

    if (!h_humidity || !h_years) {
        printf("Error: Failed to allocate host memory for trend analysis\n");
        cudaFree(d_records);
        cudaFree(d_humidity);
        cudaFree(d_years);
        if (h_humidity) free(h_humidity);
        if (h_years) free(h_years);
        return stats;
    }

    err = cudaMemcpy(h_humidity, d_humidity, count * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy humidity data back to host: %s\n", cudaGetErrorString(err));
        cudaFree(d_records);
        cudaFree(d_humidity);
        cudaFree(d_years);
        free(h_humidity);
        free(h_years);
        return stats;
    }

    err = cudaMemcpy(h_years, d_years, count * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy years data back to host: %s\n", cudaGetErrorString(err));
        cudaFree(d_records);
        cudaFree(d_humidity);
        cudaFree(d_years);
        free(h_humidity);
        free(h_years);
        return stats;
    }

    // Calculate trend (CPU - complex logic) - Fixed division by zero
    float sum_xy = 0, sum_x = 0, sum_y = 0, sum_x2 = 0;
    int min_year = h_years[0], max_year = h_years[0];  // Use device data, not original records

    for (long i = 0; i < count; i++) {
        sum_x += h_years[i];
        sum_y += h_humidity[i];
        sum_xy += h_years[i] * h_humidity[i];
        sum_x2 += h_years[i] * h_years[i];

        if (h_years[i] < min_year) min_year = h_years[i];
        if (h_years[i] > max_year) max_year = h_years[i];
    }

    long n = count;
    float denominator = (n * sum_x2 - sum_x * sum_x);
    float slope = 0.0f;
    if (fabsf(denominator) > 1e-10f) {  // Avoid division by zero
        slope = (n * sum_xy - sum_x * sum_y) / denominator;
    }
    stats.humidity_trend = slope;

    // THRUST TRANSFORM-REDUCE: Parallel variance calculation
    float variance_sum = thrust::transform_reduce(
        d_humidity_ptr, d_humidity_ptr + count,
        [mean_humidity = stats.mean_humidity] __device__ (float humidity) {
            float diff = humidity - mean_humidity;
            return diff * diff;
        },
        0.0f,
        thrust::plus<float>()
    );
    stats.humidity_variance = sqrtf(variance_sum / count);

    // Find best/worst years
    float* yearly_humidities = calloc(max_year - min_year + 1, sizeof(float));
    int* yearly_counts = calloc(max_year - min_year + 1, sizeof(int));

    for (long i = 0; i < count; i++) {
        int idx = h_years[i] - min_year;
        yearly_humidities[idx] += h_humidity[i];
        yearly_counts[idx]++;
    }

    float max_avg = -999, min_avg = 999;
    for (int year = min_year; year <= max_year; year++) {
        int idx = year - min_year;
        if (yearly_counts[idx] > 0) {
            float avg = yearly_humidities[idx] / yearly_counts[idx];
            if (avg > max_avg) {
                max_avg = avg;
                stats.best_year = year;
            }
            if (avg < min_avg) {
                min_avg = avg;
                stats.worst_year = year;
            }
        }
    }

    // Cleanup
    cudaFree(d_records);
    cudaFree(d_humidity);
    cudaFree(d_years);
    free(h_humidity);
    free(h_years);
    free(yearly_humidities);
    free(yearly_counts);

    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CUDA processing time: %.3f seconds\n", elapsed);

    return stats;
}

void printResults(HumidityStats stats, double elapsed) {
    printf("\n=== CUDA PARALLEL RESULTS ===\n");
    printf("Records processed: %ld\n", stats.record_count);
    printf("Processing time: %.3f seconds\n", elapsed);
    printf("Performance: %.0f records/second\n", stats.record_count / elapsed);
    printf("\n📊 HUMIDITY ANALYSIS RESULTS:\n");
    printf("• Mean Humidity: %.1f%%\n", stats.mean_humidity);
    printf("• Humidity Range: %.1f%% to %.1f%%\n", stats.min_humidity, stats.max_humidity);
    printf("• Humidity Volatility: %.1f%% (std dev)\n", stats.humidity_variance);

    if (stats.humidity_trend > 0) {
        printf("• 📈 Humidity Trend: +%.3f%% per year (INCREASING)\n", stats.humidity_trend);
    } else if (stats.humidity_trend < 0) {
        printf("• 📉 Humidity Trend: %.3f%% per year (DECREASING)\n", stats.humidity_trend);
    } else {
        printf("• 📊 Humidity Trend: Stable\n");
    }

    printf("• Best Year: %d\n", stats.best_year);
    printf("• Worst Year: %d\n", stats.worst_year);

    printf("\n🎯 HUMIDITY Comfort Assessment:\n");
    if (stats.mean_humidity > 70 && stats.mean_humidity < 80) {
        printf("• Status: COMFORTABLE humidity levels\n");
        printf("• Recommendation: Maintain current conditions\n");
    } else if (stats.mean_humidity >= 60 && stats.mean_humidity <= 85) {
        printf("• Status: ACCEPTABLE humidity levels\n");
        printf("• Recommendation: Minor adjustments suggested\n");
    } else {
        printf("• Status: UNCOMFORTABLE humidity levels\n");
        printf("• Recommendation: Significant adjustments needed\n");
    }

    printf("=================================\n\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <humidity_data.csv>\n", argv[0]);
        return 1;
    }

    printf("CUDA Parallel Humidity Analysis - THRUST SCAN/REDUCE\n");
    printf("===================================================\n");

    clock_t total_start = clock();

    long count;
    HumidityRecord* records = readHumidityCSV(argv[1], &count);
    if (!records) return 1;

    HumidityStats stats = analyzeParallel(records, count);

    clock_t total_end = clock();
    double total_elapsed = ((double)(total_end - total_start)) / CLOCKS_PER_SEC;

    printResults(stats, total_elapsed);

    free(records);
    return 0;
}