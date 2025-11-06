#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_CITIES 50
#define MAX_CITY_NAME_LENGTH 50
#define MAX_LINE_LENGTH 2048
#define MAX_RECORDS_CAP 100000

#ifdef _WIN32
  #define STRTOK(str, delim, saveptr) strtok_s((str), (delim), (saveptr))
#else
  #define STRTOK(str, delim, saveptr) strtok_r((str), (delim), (saveptr))
#endif

// ---------- CUDA error check ----------
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ---------- atomicAdd helpers ----------
// รองรับการ์ด < sm_60 สำหรับ double
#if __CUDA_ARCH__ < 600 && defined(__CUDA_ARCH__)
__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* addr_as_ull =
        reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + val;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(sum));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#define ATOMIC_ADD_D(addr, v) atomicAdd_double((addr), (v))
#else
#define ATOMIC_ADD_D(addr, v) atomicAdd((addr), (v))
#endif

// ใช้กับ long long โดย cast เป็น unsigned long long*
#define ATOMIC_ADD_LL(addr_ll, v_ll) \
    atomicAdd(reinterpret_cast<unsigned long long*>(addr_ll), static_cast<unsigned long long>(v_ll))

typedef struct {
    char city_name[MAX_CITY_NAME_LENGTH];
    double dri_value;
    double avg_humidity;
    double trend_slope;
    long long dry_days;
    long total_records;
    int data_column;
} CityDroughtRisk;

__device__ __forceinline__ double clamp01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

// ---------- Warp reduce helpers ----------
__inline__ __device__ double warpReduceSumD(double v){
    for(int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__inline__ __device__ long long warpReduceSumLL(long long v){
    for(int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// ========== Phase 1: Accumulate (multi-block per city) ==========
__global__ void accumulate_city_stats(
    const float* __restrict__ humidity, // [num_cities * stride]
    const int*   __restrict__ counts,   // [num_cities]
    double* __restrict__ sum_out,       // [num_cities]
    double* __restrict__ absdiff_out,   // [num_cities]
    double* __restrict__ first_out,     // [num_cities]
    double* __restrict__ second_out,    // [num_cities]
    long long* __restrict__ dry_out,    // [num_cities]
    int stride, int num_cities, int kBlocksPerCity
){
    int global_block = blockIdx.x;
    int city = global_block % num_cities;
    int n = counts[city];
    if (n <= 0) return;

    const float* base = humidity + city * stride;
    int tid = threadIdx.x;
    int block_id_for_city = global_block / num_cities;
    int cityWideStride = blockDim.x * kBlocksPerCity;

    double sum=0.0, absdiff=0.0, first=0.0, second=0.0;
    long long dry=0;

    int half = n/2;
    bool do_trend = (n > 100);

    for (int i = tid + block_id_for_city * blockDim.x; i < n; i += cityWideStride) {
        float h = __ldg(&base[i]);
        sum += h;
        if (h < 30.0f) dry++;
        if (i > 0) {
            float prev = __ldg(&base[i - 1]);
            absdiff += fabsf(h - prev);
        }
        if (do_trend) {
            if (i < half) first += h;
            else          second += h;
        }
    }

    // warp reduce
    sum     = warpReduceSumD(sum);
    absdiff = warpReduceSumD(absdiff);
    first   = warpReduceSumD(first);
    second  = warpReduceSumD(second);
    dry     = warpReduceSumLL(dry);

    __shared__ double s_sum[32], s_abs[32], s_fst[32], s_snd[32];
    __shared__ long long s_dry[32];

    if ((tid & 31) == 0) {
        int wid = tid >> 5;
        s_sum[wid] = sum;
        s_abs[wid] = absdiff;
        s_fst[wid] = first;
        s_snd[wid] = second;
        s_dry[wid] = dry;
    }
    __syncthreads();

    if (tid < 32) {
        int limit = blockDim.x >> 5;
        double bsum=0.0, babs=0.0, bfst=0.0, bsnd=0.0;
        long long bdry=0;
        #pragma unroll
        for (int w=0; w<limit; ++w) {
            bsum += s_sum[w];
            babs += s_abs[w];
            bfst += s_fst[w];
            bsnd += s_snd[w];
            bdry += s_dry[w];
        }
        if (tid == 0) {
            ATOMIC_ADD_D(&sum_out[city],     bsum);
            ATOMIC_ADD_D(&absdiff_out[city], babs);
            ATOMIC_ADD_D(&first_out[city],   bfst);
            ATOMIC_ADD_D(&second_out[city],  bsnd);
            ATOMIC_ADD_LL(&dry_out[city],    bdry);  // signed -> unsigned cast
        }
    }
}

// ========== Phase 2: Finalize per city ==========
__global__ void finalize_city_stats(
    const double* __restrict__ sum_in,
    const double* __restrict__ absdiff_in,
    const double* __restrict__ first_in,
    const double* __restrict__ second_in,
    const long long* __restrict__ dry_in,
    const int* __restrict__ counts,
    double* __restrict__ avg_out,
    double* __restrict__ slope_out,
    double* __restrict__ vol_out,
    double* __restrict__ dri_out,
    int num_cities
){
    int city = blockIdx.x * blockDim.x + threadIdx.x;
    if (city >= num_cities) return;

    int n = counts[city];
    if (n <= 0) {
        avg_out[city]=0.0; slope_out[city]=0.0; vol_out[city]=0.0; dri_out[city]=0.0;
        return;
    }

    double avg = sum_in[city] / (double)n;
    double vol = (n > 1) ? (absdiff_in[city] / (double)(n - 1)) : 0.0;

    int half = n/2;
    double slope = 0.0;
    if (n > 100 && half > 0 && (n - half) > 0) {
        double first_avg  = first_in[city]  / (double)half;
        double second_avg = second_in[city] / (double)(n - half);
        slope = (second_avg - first_avg) / (double)half * 100.0;
    }

    double HFI = (double)dry_in[city] / (double)n;
    double HSI = (1.0 - (avg / 100.0)) * 0.6;
    double HTI = (slope < 0.0) ? (-slope / 2.0) : 0.0;
    double HVI = vol / 100.0;

    if (HSI < 0.0) HSI = 0.0; if (HSI > 1.0) HSI = 1.0;
    if (HTI < 0.0) HTI = 0.0; if (HTI > 1.0) HTI = 1.0;
    if (HVI < 0.0) HVI = 0.0; if (HVI > 1.0) HVI = 1.0;

    double DRI = 0.35*HFI + 0.30*HSI + 0.20*HTI + 0.15*HVI;
    if (DRI < 0.0) DRI = 0.0; if (DRI > 1.0) DRI = 1.0;

    avg_out[city]   = avg;
    slope_out[city] = slope;
    vol_out[city]   = vol;
    dri_out[city]   = DRI;
}

// ---------- Host globals ----------
CityDroughtRisk cities[MAX_CITIES];
int num_cities = 0;
int total_records = 0;

static inline void rtrim(char* s) {
    int n = (int)strlen(s);
    while (n > 0) {
        char c = s[n - 1];
        if (c=='\n'||c=='\r'||c==' '||c=='\t') s[--n]='\0'; else break;
    }
}
static inline void ltrim(char* s) {
    int i=0; while (s[i]==' '||s[i]=='\t') i++;
    if (i>0) memmove(s, s+i, strlen(s)-i+1);
}
static inline void trim(char* s) {
    rtrim(s); ltrim(s);
    if (s[0]=='"'||s[0]=='\'') {
        size_t len=strlen(s);
        if (len>=2 && s[len-1]==s[0]) { s[len-1]='\0'; memmove(s, s+1, len); }
    }
}

int compareDRI(const void* a, const void* b) {
    const CityDroughtRisk* A=(const CityDroughtRisk*)a;
    const CityDroughtRisk* B=(const CityDroughtRisk*)b;
    if (B->dri_value > A->dri_value) return 1;
    if (B->dri_value < A->dri_value) return -1;
    return strcmp(A->city_name, B->city_name);
}

int parseCityHeaders(const char* header_line) {
    char line_copy[MAX_LINE_LENGTH];
    strncpy(line_copy, header_line, sizeof(line_copy));
    line_copy[sizeof(line_copy)-1]='\0';

    char* saveptr=NULL;
    char* token = STRTOK(line_copy, ",", &saveptr);
    int column = 1; // หลัง datetime

    num_cities = 0;
    while ((token = STRTOK(NULL, ",", &saveptr)) != NULL && num_cities < MAX_CITIES) {
        trim(token);
        if (strlen(token)>0) {
            strncpy(cities[num_cities].city_name, token, MAX_CITY_NAME_LENGTH-1);
            cities[num_cities].city_name[MAX_CITY_NAME_LENGTH-1]='\0';
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

void loadMultiCityData(const char* filepath) {
    FILE* f=fopen(filepath,"r");
    if(!f){ fprintf(stderr,"Error: Cannot open %s\n", filepath); exit(EXIT_FAILURE); }
    char line[MAX_LINE_LENGTH];
    if (!fgets(line, sizeof(line), f)) {
        fprintf(stderr,"Error: empty file %s\n", filepath);
        fclose(f); exit(EXIT_FAILURE);
    }
    parseCityHeaders(line);
    fclose(f);
}

// ========== Fast path with packing + multi-block ==========
void processAllDataGPU(const char* filepath) {
    FILE* file=fopen(filepath,"r");
    if(!file){ fprintf(stderr,"Error: Cannot open %s\n", filepath); exit(EXIT_FAILURE); }

    char line[MAX_LINE_LENGTH];
    if(!fgets(line,sizeof(line),file)){ fprintf(stderr,"Error: cannot read header again\n"); fclose(file); exit(EXIT_FAILURE); }

    // packed per city
    float** city_data = (float**)malloc(num_cities * sizeof(float*));
    int* city_write = (int*)calloc(num_cities, sizeof(int));
    if(!city_data||!city_write){ fprintf(stderr,"host alloc fail\n"); exit(EXIT_FAILURE); }
    for(int i=0;i<num_cities;i++){
        city_data[i]=(float*)malloc(MAX_RECORDS_CAP*sizeof(float));
        if(!city_data[i]){ fprintf(stderr,"host alloc city %d fail\n",i); exit(EXIT_FAILURE); }
    }

    while(fgets(line,sizeof(line),file)) {
        char* saveptr=NULL;
        char* token = STRTOK(line, ",", &saveptr);
        int column=0; (void)token; column++; // skip datetime

        while ((token = STRTOK(NULL, ",", &saveptr)) != NULL) {
            if (column >= (num_cities+1)) break;
            trim(token);
            float h=(float)atof(token);

            int city_index=-1;
            for(int i=0;i<num_cities;i++){
                if(cities[i].data_column==column){ city_index=i; break; }
            }
            if (city_index>=0 && h>0.0f && h<=100.0f) {
                int w=city_write[city_index];
                if(w<MAX_RECORDS_CAP){ city_data[city_index][w]=h; city_write[city_index]=w+1; }
            }
            column++;
        }
    }
    fclose(file);

    int max_records=0;
    for(int i=0;i<num_cities;i++){
        cities[i].total_records = city_write[i];
        if(city_write[i] > max_records) max_records = city_write[i];
    }
    total_records = max_records;

    size_t flatN = (size_t)num_cities * (size_t)max_records;

    // pinned host memory → copy เร็วขึ้น
    float* flattened_data = NULL;
    CHECK_CUDA(cudaHostAlloc((void**)&flattened_data, flatN*sizeof(float), cudaHostAllocDefault));
    memset(flattened_data, 0, flatN*sizeof(float));
    for(int c=0;c<num_cities;c++){
        int n=city_write[c];
        if(n>0){
            memcpy(flattened_data + (size_t)c*max_records, city_data[c], (size_t)n*sizeof(float));
        }
    }

    int* h_counts=(int*)malloc(num_cities*sizeof(int));
    for(int i=0;i<num_cities;i++) h_counts[i]=city_write[i];

    // ----- Device buffers -----
    int *d_counts=NULL;
    float *d_hum=NULL;
    double *d_sum=NULL, *d_abs=NULL, *d_first=NULL, *d_second=NULL;
    long long *d_dry=NULL;
    double *d_avg=NULL, *d_slope=NULL, *d_vol=NULL, *d_dri=NULL;

    CHECK_CUDA(cudaMalloc(&d_counts, num_cities*sizeof(int)));
    if(max_records>0) CHECK_CUDA(cudaMalloc(&d_hum, flatN*sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_sum,    num_cities*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_abs,    num_cities*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_first,  num_cities*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_second, num_cities*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_dry,    num_cities*sizeof(long long)));

    CHECK_CUDA(cudaMalloc(&d_avg,   num_cities*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_slope, num_cities*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_vol,   num_cities*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_dri,   num_cities*sizeof(double)));

    CHECK_CUDA(cudaFuncSetCacheConfig(accumulate_city_stats, cudaFuncCachePreferL1));

    CHECK_CUDA(cudaMemcpy(d_counts, h_counts, num_cities*sizeof(int), cudaMemcpyHostToDevice));
    if(max_records>0)
        CHECK_CUDA(cudaMemcpy(d_hum, flattened_data, flatN*sizeof(float), cudaMemcpyHostToDevice));

    // clear accumulators
    CHECK_CUDA(cudaMemset(d_sum,    0, num_cities*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_abs,    0, num_cities*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_first,  0, num_cities*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_second, 0, num_cities*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_dry,    0, num_cities*sizeof(long long)));

    // -------- Phase 1: accumulate --------
    int threads = 256;
    int kBlocksPerCity = 8;                // ปรับเป็น 4/8/16 แล้ววัด
    int blocks = (num_cities>0) ? (num_cities * kBlocksPerCity) : 0;

    if (max_records>0 && blocks>0) {
        accumulate_city_stats<<<blocks, threads>>>(
            d_hum, d_counts, d_sum, d_abs, d_first, d_second, d_dry,
            max_records, num_cities, kBlocksPerCity
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // -------- Phase 2: finalize --------
    int fThreads = 128;
    int fBlocks  = (num_cities + fThreads - 1) / fThreads;
    finalize_city_stats<<<fBlocks, fThreads>>>(
        d_sum, d_abs, d_first, d_second, d_dry, d_counts,
        d_avg, d_slope, d_vol, d_dri, num_cities
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // -------- Copy back & update --------
    double* h_avg   = (double*)malloc(num_cities*sizeof(double));
    double* h_slope = (double*)malloc(num_cities*sizeof(double));
    double* h_vol   = (double*)malloc(num_cities*sizeof(double));
    double* h_dri   = (double*)malloc(num_cities*sizeof(double));
    long long* h_dry= (long long*)malloc(num_cities*sizeof(long long));

    CHECK_CUDA(cudaMemcpy(h_avg,   d_avg,   num_cities*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_slope, d_slope, num_cities*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_vol,   d_vol,   num_cities*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dri,   d_dri,   num_cities*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dry,   d_dry,   num_cities*sizeof(long long), cudaMemcpyDeviceToHost));

    for(int i=0;i<num_cities;i++){
        cities[i].avg_humidity = h_avg[i];
        cities[i].dry_days     = h_dry[i];
        cities[i].trend_slope  = h_slope[i];
        cities[i].dri_value    = h_dri[i];
    }
    qsort(cities, num_cities, sizeof(CityDroughtRisk), compareDRI);

    // -------- Free --------
    CHECK_CUDA(cudaFree(d_counts));
    if(d_hum) CHECK_CUDA(cudaFree(d_hum));
    CHECK_CUDA(cudaFree(d_sum));
    CHECK_CUDA(cudaFree(d_abs));
    CHECK_CUDA(cudaFree(d_first));
    CHECK_CUDA(cudaFree(d_second));
    CHECK_CUDA(cudaFree(d_dry));
    CHECK_CUDA(cudaFree(d_avg));
    CHECK_CUDA(cudaFree(d_slope));
    CHECK_CUDA(cudaFree(d_vol));
    CHECK_CUDA(cudaFree(d_dri));

    CHECK_CUDA(cudaFreeHost(flattened_data));
    for(int i=0;i<num_cities;i++) free(city_data[i]);
    free(city_data);
    free(city_write);
    free(h_counts);
    free(h_avg); free(h_slope); free(h_vol); free(h_dri); free(h_dry);
}

// ---------- Print ----------
void printTop3Cities(double process_time) {
    printf("Dataset (max valid per city): %d records\n", total_records);
    printf("Top 3 Drought Risk Cities:\n");
    for (int i = 0; i < 3 && i < num_cities; i++) {
        printf("%d. %s (DRI: %.3f)\n", i + 1, cities[i].city_name, cities[i].dri_value);
    }
    printf("Time: %.3f seconds\n", process_time);
}

// ---------- Main ----------
int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <humidity_data.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    clock_t start = clock();

    loadMultiCityData(argv[1]);
    processAllDataGPU(argv[1]);

    clock_t end = clock();
    double t = (double)(end - start) / CLOCKS_PER_SEC;

    printTop3Cities(t);
    return EXIT_SUCCESS;
}
