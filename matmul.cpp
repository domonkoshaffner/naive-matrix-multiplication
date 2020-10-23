#include <CL/cl2.hpp>

#include <chrono>
#include <numeric>
#include <iterator>

#include <vector>       // std::vector
#include <exception>    // std::runtime_error, std::exception
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <random>       // std::default_random_engine, std::uniform_real_distribution
#include <algorithm>    // std::transform
#include <cstdlib>      // EXIT_FAILURE
#include <iomanip>

int main()
{
    try
    {
        // Checking the device info
        cl::CommandQueue queue = cl::CommandQueue::getDefault();
        cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();

        // Load program source
        std::ifstream source_file{ "C:/Users/haffn/Desktop/MSc-III/GPU-II/Projects/matmul2/matmul.cl" };
        if (!source_file.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "matmul.cl" };

        // Creating the program
        cl::Program program{ std::string{ std::istreambuf_iterator<char>{ source_file }, std::istreambuf_iterator<char>{} } };
        program.build({ device });

        // Creating the kernel
        auto matmul = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "matmul");

        // Init computation
        constexpr int size = 512;

        // Creating matrices for the calculations
        std::vector<double> A(size*size), B(size*size), result_cpu(size*size), result_gpu(size*size);

        // Algorithm for uniform distribution between -1 and 1
        std::random_device rnd_device;
        std::mt19937 mersenne_engine(rnd_device());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        auto gen = [&]() { return dist(mersenne_engine); };

        // Filling up the A and B matrices with random uniform distribution 
        std::generate(A.begin(), A.end(), gen );
        std::generate(B.begin(), B.end(), gen );
        // Filling up the results matrices with zeros
        std::fill(result_cpu.begin(), result_cpu.end(), 0.0f);
        std::fill(result_gpu.begin(), result_gpu.end(), 0.0f);

// ############################################################################
//naive implementation:

        auto time_cpu0 = std::chrono::high_resolution_clock::now();
        for(int i=0; i<size; ++i)
        {
            for(int j=0; j<size; ++j)
            {
                auto acc = 0.0;
                for(int k=0; k<size; ++k)
                {
                    acc += A[i*size+k] * B[k*size+j];
                }
                result_cpu[i*size+j] = acc;
            }
        }
        auto time_cpu1 = std::chrono::high_resolution_clock::now();
        auto time_difference_cpu = std::chrono::duration_cast<std::chrono::microseconds>(time_cpu1-time_cpu0).count()/1000.0;

// ############################################################################   
// Creating buffers, copying to the GPU, measuring time 

        // Creating buffers from the vectors
        cl::Buffer buf_A{ std::begin(A), std::end(A), true };
        cl::Buffer buf_B{ std::begin(B), std::end(B), true };    
        cl::Buffer buf_result_gpu{ std::begin(result_gpu), std::end(result_gpu), false };

        // Starting the clock
        auto time_gpu0 = std::chrono::high_resolution_clock::now();

        // Explicit (blocking) dispatch of data before launch
        cl::copy(queue, std::begin(A), std::end(A), buf_A);
        cl::copy(queue, std::begin(B), std::end(B), buf_B);
        cl::copy(queue, std::begin(result_gpu), std::end(result_gpu), buf_result_gpu);

        // Launch kernels
        matmul(cl::EnqueueArgs{queue, cl::NDRange{ size*size } }, buf_A, buf_B, buf_result_gpu, size);

        cl::finish();

        // (Blocking) fetch of results
        cl::copy(queue, buf_A, std::begin(A), std::end(A));
        cl::copy(queue, buf_B, std::begin(B), std::end(B));
        cl::copy(queue, buf_result_gpu, std::begin(result_gpu), std::end(result_gpu));

        // Stopping the clock and calculating the ellapsed time 
        auto time_gpu1 = std::chrono::high_resolution_clock::now();
        auto time_difference_gpu = std::chrono::duration_cast<std::chrono::microseconds>(time_gpu1-time_gpu0).count()/1000.0;   


// ############################################################################ 
// Plotting the results

        std::cout << std::endl << "The computational time for a " << size << "*" << size << " matrix multiplication on the CPU: " << time_difference_cpu  << " milisec.";
        std::cout << std::endl << "The computational time for a " << size << "*" << size << " matrix multiplication on the GPU: " << time_difference_gpu  << " milisec." << std::endl;
        std::cout << std::endl << "The GPU proves to be " << std::fixed << std::setprecision(0) << time_difference_cpu/time_difference_gpu << " times faster." << std::endl;

// ############################################################################

    }
    catch (cl::BuildError& error) // If kernel failed to build
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

        for (const auto& log : error.getBuildLog())
        {
            std::cerr <<
                "\tBuild log for device: " <<
                log.first.getInfo<CL_DEVICE_NAME>() <<
                std::endl << std::endl <<
                log.second <<
                std::endl << std::endl;
        }

        std::exit(error.err());
    }
    catch (cl::Error& error) // If any OpenCL error occurs
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
        std::exit(error.err());
    }
    catch (std::exception& error) // If STL/CRT error occurs
    {
        std::cerr << error.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
