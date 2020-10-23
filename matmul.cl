__kernel void matmul(__global float* A, 
                      __global float* B, 
                      __global float* C, 
                      int size)
{
  
   int thx = get_global_id(0); 
   int thy = get_global_id(1);

   float acc = 0.0;
   for (int i = 0; i < size; ++i)
   {
      acc += A[thy * size + i] * B[i * size + thx];
   }
 
   C[thy * size + thx] = acc;
}
