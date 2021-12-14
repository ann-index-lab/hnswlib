#include <sys/stat.h>

#include <iostream>
#include <unordered_set>

#include "hnswlib/hnswlib.h"

using namespace hnswlib;

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
static float test_approx(
    unsigned char* massQ, size_t vecsize, size_t qsize,
    HierarchicalNSW<float>& appr_alg, size_t vecdim,
    std::vector<std::priority_queue<std::pair<float, labeltype>>>& answers,
    size_t k);
static void test_vs_recall(
    unsigned char* massQ, size_t vecsize, size_t qsize,
    HierarchicalNSW<float>& appr_alg, size_t vecdim,
    std::vector<std::priority_queue<std::pair<float, labeltype>>>& answers,
    size_t k);

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

  L2Space l2space(vecdim);
  HierarchicalNSW<float>* appr_alg;

  appr_alg = new HierarchicalNSW<float>(&l2space, vecsize, M, efConstruction);
#pragma omp parallel for
  for (int i = 0; i < vecsize; i++) {
    appr_alg->addPoint((void*)(xt + i * vecdim), (size_t)i);
  }
  std::cout << "end of adding alll point" << std::endl;
  std::vector<std::priority_queue<std::pair<float, labeltype>>> answers;
  (std::vector<std::priority_queue<std::pair<float, labeltype>>>(qsize))
      .swap(answers);
  for (int i = 0; i < qsize; i++) {
    for (int j = 0; j < k; j++) {
      answers[i].emplace(0.0f, static_cast<size_t>(gt[i * k + j]));
    }
  }
  test_vs_recall((unsigned char*)xq, vecsize, qsize, *appr_alg, vecdim, answers,
                 k);

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

static float test_approx(
    unsigned char* massQ, size_t vecsize, size_t qsize,
    HierarchicalNSW<float>& appr_alg, size_t vecdim,
    std::vector<std::priority_queue<std::pair<float, labeltype>>>& answers,
    size_t k) {
  size_t correct = 0;
  size_t total = 0;
  // uncomment to test in parallel mode:
  //#pragma omp parallel for
  for (int i = 0; i < qsize; i++) {
    // std::cout<<"qsize i"<<i<<std::endl;
    std::priority_queue<std::pair<float, labeltype>> result =
        appr_alg.searchKnn((float*)(massQ) + vecdim * i, k);
    std::priority_queue<std::pair<float, labeltype>> gt(answers[i]);
    std::unordered_set<labeltype> g;
    total += gt.size();

    // cout<<"gt "<<gt.top().second<<std::endl;
    while (gt.size()) {
      g.insert(gt.top().second);

      gt.pop();
    }

    //  cout<<"result"<<result.top().second<<std::endl;
    while (result.size()) {
      if (g.find(result.top().second) != g.end()) {
        correct++;
      } else {
      }
      //  cout<<"result"<<result.top().second<<std::endl;
      result.pop();
    }
  }
  return 1.0f * correct / total;
}

static void test_vs_recall(
    unsigned char* massQ, size_t vecsize, size_t qsize,
    HierarchicalNSW<float>& appr_alg, size_t vecdim,
    std::vector<std::priority_queue<std::pair<float, labeltype>>>& answers,
    size_t k) {
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
  for (size_t ef : efs) {
    appr_alg.setEf(ef);
    StopW stopw = StopW();
    std::cout << "search efs" << ef << std::endl;
    float recall =
        test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
    float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;

    std::cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
    if (recall > 1.0) {
      std::cout << recall << "\t" << time_us_per_query << " us\n";
      break;
    }
  }
}
