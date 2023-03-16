#define R_INDEX 0
#define G_INDEX 1
#define B_INDEX 2

__global__ void background_mask(unsigned char* input_image, unsigned char* output_image, int width, int height)
{
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (col < width && row < height)
	{
		int pixel_index = (row * width + col) * 3; //her rgb pixelin indexi
		//printf("%d/", input_image[pixel_index]);
		unsigned char r = input_image[pixel_index + R_INDEX];
		unsigned char g = input_image[pixel_index + G_INDEX];
		unsigned char b = input_image[pixel_index + B_INDEX];
		float luminance = 0.299f * r + 0.587f * g + 0.114f * b;

		//output_image[pixel_index + R_INDEX] = luminance > 90 ? 255 : 0;
		//output_image[pixel_index + G_INDEX] = luminance > 90 ? 255 : 0;
		//output_image[pixel_index + B_INDEX] = luminance > 90 ? 255 : 0;

		output_image[pixel_index] = input_image[pixel_index];
		output_image[pixel_index + 1] = input_image[pixel_index+1];
		output_image[pixel_index + 2] = input_image[pixel_index+2];

	}
	else
	{
		//printf("\nthreadIdx.x = %d threadIdy.y = %d blockIdx.x = %d blockIdx.y = %d blockDim.x = %d blockDim.y = %d gridDim.x= %d gridDim.y = %d \n ", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
		// width height deðerinden büyük olan iþ parçacýklarý
	}

}