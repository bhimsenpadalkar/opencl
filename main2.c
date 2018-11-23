
#define PROGRAM_FILE "weasel.cl"
#define KERNEL_FUNC "mutate"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <CL/opencl.h>

#define NUM_OF_COPIES 10

cl_program get_program(cl_context context) {
   FILE *program_handle;
   cl_program program;
   char *program_buffer, *program_log;
   size_t program_size, log_size;

   program_handle = fopen(PROGRAM_FILE, "r");
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size,
      program_handle);
   fclose(program_handle);
   program = clCreateProgramWithSource(context, 1,
         (const char**)&program_buffer, &program_size, NULL);
      free(program_buffer);
   return program;
}

int max_score_sentence(int scores[NUM_OF_COPIES]) {
   int max_id = 0;
   for(int i=0; i<NUM_OF_COPIES ; i++) {
      if(scores[i] > scores[max_id]) max_id = i;
   }
   return max_id;
}

int main() {
   cl_platform_id platform;
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_int err;
   cl_program program;
   cl_kernel kernel;
   size_t work_units_per_kernel;
   cl_mem sentence_buff, target_sentence_buff, mutated_sentences_buff, res_buff, score_buff, size_buff;

   clGetPlatformIDs(1, &platform, NULL);
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1,
      &device, NULL);
   context = clCreateContext(NULL, 1, &device, NULL,
      NULL, &err);

   program = get_program(context);
   clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

   int scores[NUM_OF_COPIES];
   int size = 28;
   char sentence[28] = "WDLTMNLT DTJBKWIRZREZLMQCO P";
   char target_sentence[] = "METHINKS IT IS LIKE A WEASEL";
   work_units_per_kernel = NUM_OF_COPIES;
   score_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * NUM_OF_COPIES, NULL, &err);
   size_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), &size, &err);
       mutated_sentences_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(char) * NUM_OF_COPIES * size, NULL, &err);

    int counter = 0;
       target_sentence_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
                CL_MEM_COPY_HOST_PTR, sizeof(target_sentence), &target_sentence, &err);

       sentence_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
            CL_MEM_COPY_HOST_PTR, sizeof(sentence), &sentence, &err);
   while(counter < 2) {
       kernel = clCreateKernel(program, KERNEL_FUNC, &err);

       char mutated_sentences[NUM_OF_COPIES][size];

       clSetKernelArg(kernel, 0, sizeof(cl_mem), &sentence_buff);
       clSetKernelArg(kernel, 1, sizeof(cl_mem), &target_sentence_buff);
       clSetKernelArg(kernel, 2, sizeof(int), &size);
       clSetKernelArg(kernel, 3, sizeof(cl_mem), &score_buff);
       clSetKernelArg(kernel, 4, sizeof(cl_mem), &mutated_sentences_buff);

       clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
          &work_units_per_kernel, NULL, 0, NULL, NULL);

       clEnqueueReadBuffer(queue, score_buff, CL_TRUE, 0,
          sizeof(scores), &scores, 0, NULL, NULL);

       clEnqueueReadBuffer(queue, mutated_sentences_buff, CL_TRUE, 0,
            sizeof(scores), &mutated_sentences, 0, NULL, NULL);

       int max_id = max_score_sentence(scores);
    for(int k=0; k<size; k++) sentence[k] = mutated_sentences[max_id][k];
       counter++;

       printf("In Loop %d %s \n", counter, mutated_sentences[max_id]);
   }

   clReleaseMemObject(sentence_buff);
   clReleaseMemObject(res_buff);
   clReleaseMemObject(size_buff);
   clReleaseMemObject(score_buff);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}