#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

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

// Global variables for multi-city data
CityDroughtRisk cities[MAX_CITIES];
int num_cities = 0;
int total_records = 0;

// Function declarations
int parseCityHeaders(const char* header_line);
void loadMultiCityData(const char* filepath);
void calculateDRIForCity(CityDroughtRisk* city, float** humidity_data, int* record_count);
void printTop3Cities(double process_time);
int compareDRI(const void* a, const void* b);

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

    // Close file for now - we'll process data more efficiently
    fclose(file);
}

// Calculate DRI for a single city
void calculateDRIForCity(CityDroughtRisk* city, float** humidity_data, int* record_count) {
    if (city->total_records == 0) return;

    // Calculate basic statistics
    double sum = 0.0;
    long dry_days = 0;

    for (int i = 0; i < city->total_records; i++) {
        float humidity = humidity_data[city->data_column][i];
        sum += humidity;
        if (humidity < 30.0f) {
            dry_days++;
        }
    }

    city->avg_humidity = sum / city->total_records;
    city->dry_days = dry_days;

    // Calculate HFI - Humidity Frequency Index
    double HFI = (double)dry_days / city->total_records;

    // Calculate HSI - Humidity Severity Index
    double avg_ratio = city->avg_humidity / 100.0;
    double HSI = (1.0 - avg_ratio) * 0.6;  // Simplified version

    // Calculate HTI - Humidity Trend Index (simplified linear trend)
    if (city->total_records > 100) {
        int first_half = city->total_records / 2;
        double first_avg = 0.0, second_avg = 0.0;

        for (int i = 0; i < first_half; i++) {
            first_avg += humidity_data[city->data_column][i];
        }
        first_avg /= first_half;

        for (int i = first_half; i < city->total_records; i++) {
            second_avg += humidity_data[city->data_column][i];
        }
        second_avg /= (city->total_records - first_half);

        city->trend_slope = (second_avg - first_avg) / (city->total_records / 2.0) * 100.0;
    }

    double HTI = (city->trend_slope < 0) ? (-city->trend_slope / 2.0) : 0.0;
    if (HTI > 1.0) HTI = 1.0;

    // Calculate HVI - Humidity Volatility Index (simplified)
    double volatility = 0.0;
    if (city->total_records > 1) {
        for (int i = 1; i < city->total_records; i++) {
            volatility += fabs(humidity_data[city->data_column][i] - humidity_data[city->data_column][i-1]);
        }
        volatility /= (city->total_records - 1);
    }
    double HVI = volatility / 100.0;
    if (HVI > 1.0) HVI = 1.0;

    // Calculate final DRI
    city->dri_value = 0.35 * HFI + 0.30 * HSI + 0.20 * HTI + 0.15 * HVI;
    if (city->dri_value > 1.0) city->dri_value = 1.0;
}

// Comparison function for sorting
int compareDRI(const void* a, const void* b) {
    CityDroughtRisk* cityA = (CityDroughtRisk*)a;
    CityDroughtRisk* cityB = (CityDroughtRisk*)b;
    if (cityB->dri_value > cityA->dri_value) return 1;
    if (cityB->dri_value < cityA->dri_value) return -1;
    return 0;
}

// Process all data and calculate DRI for all cities
void processAllData(const char* filepath) {
    FILE* file = fopen(filepath, "r");
    if (!file) return;

    char line[MAX_LINE_LENGTH];

    // Skip header
    fgets(line, sizeof(line), file);

    // Allocate memory for humidity data (dynamic 2D array)
    float** humidity_data = (float**)malloc((num_cities + 5) * sizeof(float*));
    for (int i = 0; i < num_cities + 5; i++) {
        humidity_data[i] = (float*)malloc(100000 * sizeof(float));  // Assuming max 100k records
    }

    int record_count = 0;

    // Read data line by line
    while (fgets(line, sizeof(line), file) && record_count < 100000) {
        char* token = strtok(line, ",");
        int column = 0;

        // Skip datetime column
        column++;

        // Read humidity values for each city
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

    // Calculate DRI for each city
    for (int i = 0; i < num_cities; i++) {
        calculateDRIForCity(&cities[i], humidity_data, &record_count);
    }

    // Sort cities by DRI
    qsort(cities, num_cities, sizeof(CityDroughtRisk), compareDRI);

    // Free memory
    for (int i = 0; i < num_cities + 5; i++) {
        free(humidity_data[i]);
    }
    free(humidity_data);
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
int program_execution(int parameter_count, char* parameter_values[]) {
    if (parameter_count != 2) {
        fprintf(stderr, "Usage: %s <humidity_data.csv>\n", parameter_values[0]);
        return EXIT_FAILURE;
    }

    clock_t start_time = clock();

    loadMultiCityData(parameter_values[1]);
    processAllData(parameter_values[1]);

    clock_t end_time = clock();
    double process_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printTop3Cities(process_time);

    return EXIT_SUCCESS;
}

// Standard main function wrapper
int main(int argc, char* argv[]) {
    return program_execution(argc, argv);
}