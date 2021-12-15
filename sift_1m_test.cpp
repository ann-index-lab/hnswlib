#include <sys/stat.h>

#include <iostream>
#include <unordered_set>

#include "hnswlib/hnswlib.h"
#include "data_divide_hnswlib/hnswlib.h"


class StopW {
  std::chrono::steady_clock::time_point time_begin;

 public:
  StopW() { time_begin = std::chrono::steady_clock::now(); }

  float getElapsedTimeMicro() {
    std::chrono::steady_clock::time_point time_end =
        std::chrono::steady_clock::now();
    return (std::chrono::duration_cast<std::chrono::microseconds>(time_end -
                                                                  time_begin)
                .count());
  }

  void reset() { time_begin = std::chrono::steady_clock::now(); }
};

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out);
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out);
static void test_normal_hnsw_in_sift1m(
        float* data_set, float* query_set, size_t dsize, size_t qsize, size_t dim,
        std::vector<std::priority_queue<std::pair<float, size_t>>>& answers,
        size_t k, int M, int efConstruction );
static void test_data_divide_hnsw_in_sift1m(
        float* data_set, float* query_set, size_t dsize, size_t qsize, size_t dim,
        std::vector<std::priority_queue<std::pair<float, size_t>>>& answers,
        size_t k, int M, int efConstruction );

int main() {
  int efConstruction = 200;
  int M = 200;

  size_t vecsize;
  size_t qsize;
  size_t vecdim;
  size_t k;

  // get data
  std::cout << "get xt data" << std::endl;
  float* xt = fvecs_read("../dataset/sift/sift_base.fvecs", &vecdim, &vecsize);
  float* xq = fvecs_read("../dataset/sift/sift_query.fvecs", &vecdim, &qsize);
  int* gt = ivecs_read("../dataset/sift/sift_groundtruth.ivecs", &k, &qsize);

  std::cout << "xt size " << vecsize << " dim" << vecdim;
  std::cout << "qt size " << qsize << " dim" << vecdim;
  std::cout << "topk " << k << std::endl;

  //set answer:
  std::vector<std::priority_queue<std::pair<float, size_t>>> answers;
  (std::vector<std::priority_queue<std::pair<float, size_t>>>(qsize))
          .swap(answers);
  for (int i = 0; i < qsize; i++) {
    for (int j = 0; j < k; j++) {
      answers[i].emplace(0.0f, static_cast<size_t>(gt[i * k + j]));
    }
  }

  std::cout<<"test sift1m in normal hnsw"<<std::endl;
  test_normal_hnsw_in_sift1m(xt, xq, vecsize, qsize, vecdim, answers, k, M, efConstruction);

  std::cout<<"test sift1m in data divide hnsw"<<std::endl;
  test_data_divide_hnsw_in_sift1m(xt, xq, vecsize, qsize, vecdim, answers, k, M, efConstruction);

  delete[] gt;
  delete[] xq;
  delete[] xt;
  return 0;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
  FILE* f = fopen(fname, "r");
  if (!f) {
    fprintf(stderr, "could not open %s\n", fname);
    perror("");
    abort();
  }
  int d;
  fread(&d, 1, sizeof(int), f);
  assert((d > 0 && d < 1000000) || !"unreasonable dimension");
  fseek(f, 0, SEEK_SET);
  struct stat st;
  fstat(fileno(f), &st);
  size_t sz = st.st_size;
  assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
  size_t n = sz / ((d + 1) * 4);

  *d_out = d;
  *n_out = n;
  float* x = new float[n * (d + 1)];
  size_t nr = fread(x, sizeof(float), n * (d + 1), f);
  assert(nr == n * (d + 1) || !"could not read whole file");

  for (size_t i = 0; i < n; i++)
    memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

  fclose(f);
  return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
  return (int*)fvecs_read(fname, d_out, n_out);
}

float recall_function(std::vector<std::priority_queue<std::pair<float, size_t>>> result,
        std::vector<std::priority_queue<std::pair<float, size_t>>> gt) {
  size_t correct = 0;
  size_t total = 0;
  std::unordered_set<size_t> g;
  for (int i = 0; i < result.size(); i++) {
    g.clear();
    total += gt.size();

    while (gt[i].size()) {
      g.insert(gt[i].top().second);

      gt[i].pop();
    }

    while (result[i].size()) {
      if (g.find(result[i].top().second) != g.end()) {
        correct++;
      } else {

      }

      result[i].pop();
    }

  }
  return 1.0f * correct / total;
}

std::vector<size_t> generate_efs_set(size_t k) {
  std::vector<size_t> efs;  // = { 10,10,10,10,10 };
  for (int i = k; i < 30; i++) {
    efs.push_back(i);
  }
  for (int i = 30; i < 100; i += 10) {
    efs.push_back(i);
  }
  for (int i = 100; i < 500; i += 40) {
    efs.push_back(i);
  }
  return efs;
}

static void test_normal_hnsw_in_sift1m(
        float* data_set, float* query_set, size_t dsize, size_t qsize, size_t dim,
        std::vector<std::priority_queue<std::pair<float, size_t>>>& answers,
        size_t k, int M, int efConstruction ) {
  hnswlib::L2Space l2space(dim);
  hnswlib::HierarchicalNSW<float>* appr_alg;

  appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, dsize, M, efConstruction);

  std::cout<<"building normal hnsw"<<std::endl;
#pragma omp parallel for
  for (int i = 0; i < dsize; i++) {
    appr_alg->addPoint((void*)(data_set + i * dim), (size_t)i);
  }

  std::cout<<"searching in hnsw"<<std::endl;

  std::vector<std::priority_queue<std::pair<float, size_t>>> result(qsize);

  std::cout<<"getting the result of normal hnsw"<<std::endl;
  auto efs = generate_efs_set(k);
  for (size_t ef : efs) {
    appr_alg->setEf(ef);
    StopW stopw = StopW();
    std::cout << "search efs" << ef << std::endl;
// uncomment to test in parallel mode:
//#pragma omp parallel for
    for (int i = 0; i < qsize; i++) {
      // std::cout<<"qsize i"<<i<<std::endl;
      result[i] = appr_alg->searchKnn(query_set + dim * i, k);
    }
    float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
    float recall = recall_function(result, answers);

    std::cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
    if (recall > 1.0) {
      std::cout << recall << "\t" << time_us_per_query << " us\n";
      break;
    }
  }
  delete appr_alg;
}

static void test_data_divide_hnsw_in_sift1m(
        float* data_set, float* query_set, size_t dsize, size_t qsize, size_t dim,
        std::vector<std::priority_queue<std::pair<float, size_t>>>& answers,
        size_t k, int M, int efConstruction ) {
    data_divide_hnswlib::L2Space l2space(dim);
    data_divide_hnswlib::HierarchicalNSW<float>* appr_alg;

    bool data_compaction = false;

    appr_alg = new data_divide_hnswlib::HierarchicalNSW<float>(&l2space, dsize, M, efConstruction, data_compaction);

    std::cout<<"building normal hnsw"<<std::endl;
#pragma omp parallel for
    for (int i = 0; i < dsize; i++) {
        appr_alg->addPoint((void*)(data_set + i * dim), (size_t)i);
    }

    std::cout<<"searching in hnsw"<<std::endl;

    std::vector<std::priority_queue<std::pair<float, size_t>>> result(qsize);

    std::cout<<"getting the result of normal hnsw"<<std::endl;
    auto efs = generate_efs_set(k);
    for (size_t ef : efs) {
        appr_alg->setEf(ef);
        StopW stopw = StopW();
        std::cout << "search efs" << ef << std::endl;
// uncomment to test in parallel mode:
//#pragma omp parallel for
        for (int i = 0; i < qsize; i++) {
            // std::cout<<"qsize i"<<i<<std::endl;
            result[i] = appr_alg->searchKnn(query_set + dim * i, k);
        }
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
        float recall = recall_function(result, answers);

        std::cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
        if (recall > 1.0) {
            std::cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
    delete appr_alg;
}

