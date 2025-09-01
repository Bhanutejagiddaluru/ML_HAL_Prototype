
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <thread>
#include <atomic>

namespace py = pybind11;

// Very simple parallel matmul to simulate "accelerator fast path".
// NOTE: Replace with real device API calls.
py::array_t<float> matmul(py::array_t<float, py::array::c_style | py::array::forcecast> a,
                          py::array_t<float, py::array::c_style | py::array::forcecast> b) {
    auto a_buf = a.request();
    auto b_buf = b.request();
    if (a_buf.ndim != 2 || b_buf.ndim != 2) throw std::runtime_error("matmul expects 2D arrays");
    ssize_t M = a_buf.shape[0];
    ssize_t K = a_buf.shape[1];
    ssize_t K2 = b_buf.shape[0];
    ssize_t N = b_buf.shape[1];
    if (K != K2) throw std::runtime_error("shape mismatch");

    auto result = py::array_t<float>({M, N});
    auto c_buf = result.request();

    const float* A = static_cast<float*>(a_buf.ptr);
    const float* B = static_cast<float*>(b_buf.ptr);
    float* C = static_cast<float*>(c_buf.ptr);

    // Zero init
    std::fill(C, C + (M * N), 0.0f);

    // Simple parallelization by rows
    int threads = std::thread::hardware_concurrency();
    if (threads <= 0) threads = 4;
    auto worker = [&](int tid) {
        ssize_t rows_per = (M + threads - 1) / threads;
        ssize_t r0 = tid * rows_per;
        ssize_t r1 = std::min<ssize_t>(M, r0 + rows_per);
        for (ssize_t i = r0; i < r1; ++i) {
            for (ssize_t k = 0; k < K; ++k) {
                float a_ik = A[i*K + k];
                const float* b_row = &B[k*N];
                float* c_row = &C[i*N];
                for (ssize_t j = 0; j < N; ++j) {
                    c_row[j] += a_ik * b_row[j];
                }
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    return result;
}

py::array_t<float> relu(py::array_t<float, py::array::c_style | py::array::forcecast> x) {
    auto buf = x.request();
    auto y = py::array_t<float>(buf.shape);
    auto ybuf = y.request();
    const float* X = static_cast<float*>(buf.ptr);
    float* Y = static_cast<float*>(ybuf.ptr);
    ssize_t total = 1;
    for (auto s : buf.shape) total *= s;
    for (ssize_t i = 0; i < total; ++i) Y[i] = X[i] > 0.0f ? X[i] : 0.0f;
    return y;
}

py::array_t<float> add_bias(py::array_t<float, py::array::c_style | py::array::forcecast> x,
                            py::array_t<float, py::array::c_style | py::array::forcecast> b) {
    auto xbuf = x.request();
    auto bbuf = b.request();
    if (bbuf.ndim != 1) throw std::runtime_error("bias must be 1D");
    if (xbuf.ndim != 2) throw std::runtime_error("x must be 2D");
    ssize_t M = xbuf.shape[0];
    ssize_t N = xbuf.shape[1];
    if (bbuf.shape[0] != N) throw std::runtime_error("bias len mismatch");

    auto y = py::array_t<float>({M, N});
    auto ybuf = y.request();
    const float* X = static_cast<float*>(xbuf.ptr);
    const float* B = static_cast<float*>(bbuf.ptr);
    float* Y = static_cast<float*>(ybuf.ptr);

    for (ssize_t i = 0; i < M; ++i) {
        for (ssize_t j = 0; j < N; ++j) {
            Y[i*N + j] = X[i*N + j] + B[j];
        }
    }
    return y;
}

PYBIND11_MODULE(hal_ext, m) {
    m.doc() = "Mock HAL fast-path (CPU parallel)";
    m.def("matmul", &matmul, "Matrix multiply (A[M,K] x B[K,N])");
    m.def("relu", &relu, "ReLU");
    m.def("add_bias", &add_bias, "Add bias");
}
