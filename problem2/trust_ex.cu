#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

struct integral_functor
{
    const double delta_x;

    integral_functor(double delta_x) : delta_x(delta_x) {}

    __host__ __device__
    double operator()(double x)
    {
        double fx = 4.0 / (1.0 + x * x);
        return fx * delta_x;
    }
};

int main()
{
    const int N = 1000000000;
    const double a = 0.0;
    const double b = 1.0;
    const double delta_x = (b - a) / N;

    // Create a device vector with N+1 elements
    thrust::device_vector<double> x(N + 1);

    // Fill the vector with values from a to b with a step of delta_x
    thrust::sequence(x.begin(), x.end(), a, delta_x);

    // Calculate the sum of f(x_i) * delta_x using transform_reduce
    double integral = thrust::transform_reduce(x.begin(), x.end(), integral_functor(delta_x), 0.0, thrust::plus<double>());

    std::cout << "Approximate integral: " << integral << std::endl;

    return 0;
}
