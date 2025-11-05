# CUDA Humidity Trend Analysis

## How to Run Programs

### Requirements
- CUDA Toolkit 11.0+
- C Compiler (gcc/clang)
- Humidity data CSV file

### Build Commands
```bash
# Compile sequential version
gcc -O3 sequential.c -o sequential -lm

# Compile CUDA version
nvcc -O3 parallel.cu -o parallel -lm
```

### Run Programs
```bash
# Run sequential CPU version
./sequential humidity.csv

# Run CUDA parallel version
./parallel humidity.csv
```

## Dataset Format
CSV file should contain year, month, and humidity columns:

Example format:
```
year,month,humidity
2018,1,75.2
2018,2,76.8
2019,1,78.3
...
```
# parallel-computing-project
# parallel-computing-project
