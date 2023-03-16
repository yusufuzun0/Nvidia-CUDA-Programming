#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#include "stb_image.h"
#include "stb_image_write.h"


#include "background_mask.cuh"
#include "median_filter.cuh"


//#include <thrust/swap.h>

#define BLOCK_SIZE 32




int main()
{
	const std::string input_filename = "C:/Dechard/Görüntüler/instabig.jpg";
	const std::string output_filename = "C:/Dechard/cikisfoto1.jpg";

	int width, height, num_channels;
	unsigned char* input_image = stbi_load(input_filename.c_str(), &width, &height, &num_channels, 3);

	if (input_image == nullptr)
	{
		std::cerr << "Hata: Fotograf Yuklenemedi." << std::endl;
		return EXIT_FAILURE;
	}

	unsigned char* bgm_input_image;
	unsigned char* bgm_output_image;

	cudaMalloc(&bgm_input_image, width * height * 3 * sizeof(unsigned char));
	cudaMalloc(&bgm_output_image, width * height * 3 * sizeof(unsigned char));

	dim3 block_size(BLOCK_SIZE * BLOCK_SIZE);
	dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

	cudaMemcpy(bgm_input_image, input_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	background_mask <<< grid_size, block_size >>> (bgm_input_image, bgm_output_image, width, height);
	cudaDeviceSynchronize();

	unsigned char* med_input_image;
	unsigned char* med_output_image;
	cudaMalloc(&med_input_image, width * height * 3 * sizeof(unsigned char));
	cudaMalloc(&med_output_image, width * height * 3 * sizeof(unsigned char));

	cudaMemcpy(med_input_image, bgm_output_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
	median_filter_rgb <<<grid_size, block_size >>> (med_input_image, med_output_image, width, height);
	cudaDeviceSynchronize();

	unsigned char* output_image = new unsigned char[width * height * 3]; // rgb olduğu için 3 bayt ile çarpılır
	cudaMemcpy(output_image, med_output_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	stbi_write_jpg(output_filename.c_str(), width, height, 3, output_image, 100);

	cudaFree(bgm_input_image);
	cudaFree(bgm_output_image);
	cudaFree(med_input_image);
	cudaFree(med_output_image);

	delete[] input_image;
	delete[] output_image;

	return EXIT_SUCCESS;



}