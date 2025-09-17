// seedbs.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <tuple>

namespace py = pybind11;

/*
 Helper:
 - Input pvals: 1D numpy array of doubles.
 - We map values to ranks (unique sorted values) and then compute
   weighted KS for each split k by maintaining left counts incrementally.
 Complexity: O(n * G) where G = number of unique values <= n.
*/

// Compute ks_statistic: returns (best_split_index, best_S).
// best_split_index is in [1..n-1] (1-based like your python k). We'll return 0-based split index (k)
// matching your Python interpretation: split at k means left pvals[0:k], right pvals[k:].
std::pair<int, double> ks_statistic_cpp(py::array_t<double, py::array::c_style | py::array::forcecast> pvals_arr) {
    auto buf = pvals_arr.unchecked<1>();
    const ssize_t n = buf.shape(0);
    if (n < 2) {
        return {-1, 0.0};
    }

    // Copy values into vector
    std::vector<double> pvals(n);
    for (ssize_t i = 0; i < n; ++i) pvals[i] = buf(i);

    // Build unique sorted grid of values (ranks)
    std::vector<double> grid = pvals;
    std::sort(grid.begin(), grid.end());
    grid.erase(std::unique(grid.begin(), grid.end()), grid.end());
    const size_t G = grid.size();

    // Map each value to rank via binary search
    std::vector<size_t> ranks(n);
    for (ssize_t i = 0; i < n; ++i) {
        ranks[i] = std::lower_bound(grid.begin(), grid.end(), pvals[i]) - grid.begin();
    }

    // total counts per rank (for right-side counts)
    std::vector<int> total_counts(G, 0);
    for (size_t r : ranks) total_counts[r]++;

    // left counts start zero
    std::vector<int> left_counts(G, 0);

    double best_S = -1.0;
    int best_k = -1;
    const double denom_const = std::pow(static_cast<double>(n), 1.5);

    // iterate split k = 1..n-1 (meaning left has size k)
    for (int k = 1; k <= (int)n - 1; ++k) {
        // include pvals[k-1] into left_counts
        size_t r = ranks[k-1];
        left_counts[r]++;

        int size_left = k;
        int size_right = n - k;
        if (size_left <= 0 || size_right <= 0) continue;

        // compute sup over grid of |Fx - Fy|
        double D = 0.0;
        for (size_t j = 0; j < G; ++j) {
            double Fx = static_cast<double>(left_counts[j]) / static_cast<double>(size_left);
            double Fy = static_cast<double>(total_counts[j] - left_counts[j]) / static_cast<double>(size_right);
            double diff = std::abs(Fx - Fy);
            if (diff > D) D = diff;
        }

        // weight = (size_left * size_right) / (n ^ 1.5)
        double weight = (static_cast<double>(size_left) * static_cast<double>(size_right)) / (denom_const);
        double S = weight * D;
        if (S > best_S) {
            best_S = S;
            best_k = k; // keep 1..n-1 semantics
        }
    }

    if (best_S < 0) best_S = 0.0;
    return {best_k, best_S};
}


// Helper to create a single block-permuted vector (like your permute_pvalues)
static std::vector<double> block_permute(const std::vector<double>& pvals, int block_size, std::mt19937 &rng) {
    int n = (int)pvals.size();
    if (block_size <= 1) {
        // simple random permutation
        std::vector<double> out = pvals;
        std::shuffle(out.begin(), out.end(), rng);
        out.resize(n);
        return out;
    }
    int max_start = n - block_size;
    if (max_start < 0) {
        // block_size > n, just shuffle
        std::vector<double> out = pvals;
        std::shuffle(out.begin(), out.end(), rng);
        out.resize(n);
        return out;
    }

    // create vector of possible starts [0..max_start]
    std::vector<int> starts(max_start + 1);
    std::iota(starts.begin(), starts.end(), 0);
    std::shuffle(starts.begin(), starts.end(), rng);

    int needed = (n + block_size - 1) / block_size; // ceil(n/block_size)
    std::vector<double> out;
    out.reserve(needed * block_size);
    for (int i = 0; i < needed && i < (int)starts.size(); ++i) {
        int s = starts[i];
        for (int j = 0; j < block_size; ++j) {
            int idx = s + j;
            if (idx >= n) break;
            out.push_back(pvals[idx]);
        }
    }
    if ((int)out.size() > n) out.resize(n);
    // If not enough (shouldn't happen), append shuffled values
    if ((int)out.size() < n) {
        std::vector<double> tmp = pvals;
        std::shuffle(tmp.begin(), tmp.end(), rng);
        for (int i = (int)out.size(); i < n; ++i) out.push_back(tmp[i - (int)out.size()]);
    }
    return out;
}


// segment_significance: compute observed ks_statistic, then permutation p-value using block permutations.
// n_jobs controls parallelization (if compiled with OpenMP and n_jobs > 1, will parallelize loop).
std::pair<int, double> segment_significance_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> pvals_arr,
    int n_permutations = 99,
    int block_size = 10,
    uint64_t seed = 0,
    int n_jobs = 1
) {
    auto buf = pvals_arr.unchecked<1>();
    const ssize_t n = buf.shape(0);
    if (n < 2) {
        return {-1, 1.0};
    }

    // copy pvals into std::vector
    std::vector<double> pvals(n);
    for (ssize_t i = 0; i < n; ++i) pvals[i] = buf(i);

    // observed
    auto obs = ks_statistic_cpp(pvals_arr);
    int obs_k = obs.first;
    double obs_S = obs.second;
    if (obs_k <= 0) {
        return {-1, 1.0};
    }

    int count_ge = 0;

    // We'll use thread-local RNG seeds
    std::mt19937 rng_base((unsigned int)(seed == 0 ? std::random_device()() : seed));

    // If n_jobs <= 1 do sequential
    if (n_jobs <= 1) {
        for (int t = 0; t < n_permutations; ++t) {
            // derive rng for this iter
            std::mt19937 rng(rng_base());
            std::vector<double> perm = block_permute(pvals, block_size, rng);
            // wrap perm into numpy-like array view for ks_statistic_cpp call:
            py::array_t<double> perm_arr((ssize_t)perm.size());
            auto ptr = (double*)perm_arr.request().ptr;
            std::copy(perm.begin(), perm.end(), ptr);
            auto res = ks_statistic_cpp(perm_arr);
            double Sperm = res.second;
            if (Sperm >= obs_S) ++count_ge;
        }
    } else {
        // parallel loop (simple OpenMP style if available)
        // We implement manual per-thread rng initialization to avoid collisions.
        int total = n_permutations;
        std::vector<int> counts_per_thread(total, 0); // will be aggregated trivially
        // We'll do a simple parallel for if OpenMP is enabled
        #ifdef _OPENMP
        #include <omp.h>
        int max_threads = omp_get_max_threads();
        // limit threads to n_jobs
        int threads = std::min(n_jobs, max_threads);
        #pragma omp parallel num_threads(threads)
        {
            unsigned int tid = (unsigned int)omp_get_thread_num();
            // thread-local RNG seeded from rng_base
            std::mt19937 rng(rng_base() + tid + 1);
            #pragma omp for schedule(static)
            for (int t = 0; t < total; ++t) {
                std::vector<double> perm = block_permute(pvals, block_size, rng);
                py::array_t<double> perm_arr((ssize_t)perm.size());
                auto ptr = (double*)perm_arr.request().ptr;
                std::copy(perm.begin(), perm.end(), ptr);
                auto res = ks_statistic_cpp(perm_arr);
                double Sperm = res.second;
                if (Sperm >= obs_S) counts_per_thread[t] = 1;
                else counts_per_thread[t] = 0;
            }
        } // end omp
        // aggregate
        for (int t = 0; t < total; ++t) count_ge += counts_per_thread[t];
        #else
        // If OpenMP not available, fall back to sequential
        for (int t = 0; t < n_permutations; ++t) {
            std::mt19937 rng(rng_base());
            std::vector<double> perm = block_permute(pvals, block_size, rng);
            py::array_t<double> perm_arr((ssize_t)perm.size());
            auto ptr = (double*)perm_arr.request().ptr;
            std::copy(perm.begin(), perm.end(), ptr);
            auto res = ks_statistic_cpp(perm_arr);
            double Sperm = res.second;
            if (Sperm >= obs_S) ++count_ge;
        }
        #endif
    }

    // add-one correction
    double p_tilde = (static_cast<double>(count_ge) + 1.0) / (static_cast<double>(n_permutations) + 1.0);

    return {obs_k, p_tilde};
}


PYBIND11_MODULE(seedbs, m) {
    m.doc() = "Fast KS-statistic and permutation significance using pybind11";

    m.def("ks_statistic", &ks_statistic_cpp, py::arg("pvals"),
          R"pbdoc(Compute weighted KS scan statistic and best split.

Returns (best_k, S_max) where best_k is an integer split index (1..n-1).
If no valid split, best_k = -1.)pbdoc");

    m.def("segment_significance", &segment_significance_cpp,
          py::arg("pvals"),
          py::arg("n_permutations") = 99,
          py::arg("block_size") = 10,
          py::arg("seed") = 0,
          py::arg("n_jobs") = 1,
          R"pbdoc(Compute observed ks_statistic and permutation p-value.

Returns (best_k, p_tilde).)pbdoc");
}
