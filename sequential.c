#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

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

HumidityStats analyzeSequential(HumidityRecord* records, long count) {
    HumidityStats stats = {0};
    stats.record_count = count;

    if (count == 0) return stats;

    clock_t start = clock();

    float sum_humidity = 0;
    float sum_xy = 0, sum_x = 0, sum_y = 0, sum_x2 = 0;
    int min_year = 9999, max_year = 0;
    bool first_valid = true;
    long valid_count = 0;

    for (long i = 0; i < count; i++) {
        HumidityRecord* r = &records[i];

        // Input validation - ensure humidity is in reasonable range
        if (r->humidity < 0.0f || r->humidity > 100.0f) {
            printf("Warning: Invalid humidity value %.1f at record %ld, skipping\n", r->humidity, i);
            continue;
        }

        sum_humidity += r->humidity;
        valid_count++;

        // Initialize min/max on first valid record
        if (first_valid) {
            stats.min_humidity = r->humidity;
            stats.max_humidity = r->humidity;
            min_year = r->year;
            max_year = r->year;
            first_valid = false;
        } else {
            if (r->humidity < stats.min_humidity) stats.min_humidity = r->humidity;
            if (r->humidity > stats.max_humidity) stats.max_humidity = r->humidity;
        }

        if (r->year < min_year) min_year = r->year;
        if (r->year > max_year) max_year = r->year;

        sum_x += r->year;
        sum_y += r->humidity;
        sum_xy += r->year * r->humidity;
        sum_x2 += r->year * r->year;
    }

    if (valid_count == 0) {
        printf("Error: No valid humidity records found\n");
        stats.mean_humidity = 0.0f;
        stats.min_humidity = 0.0f;
        stats.max_humidity = 0.0f;
        stats.humidity_trend = 0.0f;
        stats.humidity_variance = 0.0f;
        stats.best_year = 0;
        stats.worst_year = 0;
        clock_t end = clock();
        double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Sequential processing time: %.3f seconds\n", elapsed);
        return stats;
    }

    stats.mean_humidity = sum_humidity / valid_count;

    long n = valid_count;  // Use valid_count for trend calculation
    float denominator = (n * sum_x2 - sum_x * sum_x);
    float slope = 0.0f;
    if (fabsf(denominator) > 1e-10f) {  // Avoid division by zero
        slope = (n * sum_xy - sum_x * sum_y) / denominator;
    }
    stats.humidity_trend = slope;

    float variance_sum = 0;
    for (long i = 0; i < count; i++) {
        if (records[i].humidity >= 0.0f && records[i].humidity <= 100.0f) {
            float diff = records[i].humidity - stats.mean_humidity;
            variance_sum += diff * diff;
        }
    }
    stats.humidity_variance = sqrtf(variance_sum / valid_count);

    int year_range = max_year - min_year + 1;
    if (year_range <= 0 || year_range > 1000) {  // Reasonable bounds check
        printf("Warning: Invalid year range %d to %d, skipping yearly analysis\n", min_year, max_year);
        stats.best_year = min_year;
        stats.worst_year = max_year;
        clock_t end = clock();
        double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Sequential processing time: %.3f seconds\n", elapsed);
        return stats;
    }

    float* yearly_humidities = calloc(year_range, sizeof(float));
    int* yearly_counts = calloc(year_range, sizeof(int));

    if (!yearly_humidities || !yearly_counts) {
        printf("Error: Memory allocation failed for yearly analysis\n");
        stats.best_year = min_year;
        stats.worst_year = max_year;
        clock_t end = clock();
        double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Sequential processing time: %.3f seconds\n", elapsed);
        return stats;
    }

    for (long i = 0; i < count; i++) {
        int idx = records[i].year - min_year;
        if (idx >= 0 && idx < year_range) {  // Bounds checking
            yearly_humidities[idx] += records[i].humidity;
            yearly_counts[idx]++;
        }
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

    free(yearly_humidities);
    free(yearly_counts);

    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Sequential processing time: %.3f seconds\n", elapsed);

    return stats;
}

void printResults(HumidityStats stats, double elapsed) {
    printf("\n=== SEQUENTIAL CPU RESULTS ===\n");
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

    printf("========================================\n\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <humidity_data.csv>\n", argv[0]);
        return 1;
    }

    printf("Sequential Humidity Analysis - OPTIMAL CPU IMPLEMENTATION\n");
    printf("==========================================================\n");

    clock_t total_start = clock();

    long count;
    HumidityRecord* records = readHumidityCSV(argv[1], &count);
    if (!records) return 1;

    HumidityStats stats = analyzeSequential(records, count);

    clock_t total_end = clock();
    double total_elapsed = ((double)(total_end - total_start)) / CLOCKS_PER_SEC;

    printResults(stats, total_elapsed);

    free(records);
    return 0;
}