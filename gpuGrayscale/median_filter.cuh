/*
__device__ unsigned char find_median(unsigned char* array, int size);
__global__ void median_filter_gray(unsigned char* input_image, unsigned char* output_image, int width, int height)
{
	int col = threadIdx.x + (blockIdx.x * blockDim.x);
	int row = threadIdx.y + (blockIdx.y * blockDim.y);

	if (col < width && row < height)
	{
		int window_radius = WINDOW_SIZE / 2;
		int window_width = 2 * window_radius + 1;

		// Allocate memory for window pixels
		unsigned char* window_pixels = new unsigned char[window_width * window_width];

		// Get window pixels
		for (int i = -window_radius; i <= window_radius; i++) {
			for (int j = -window_radius; j <= window_radius; j++) {
				int current_row = row + i;
				int current_col = col + j;

				if (current_row >= 0 && current_row < height && current_col >= 0 && current_col < width) {
					window_pixels[(i + window_radius) * window_width + (j + window_radius)] = input_image[current_row * width + current_col];
				}
				else {
					window_pixels[(i + window_radius) * window_width + (j + window_radius)] = 0;
				}
			}
		}

		// Get median pixel value
		output_image[row * width + col] = find_median(window_pixels, window_width * window_width);

		// Free memory
		delete[] window_pixels;
	}
}
__device__ unsigned char find_median(unsigned char* array, int size)
{
	 //Pencere piksellerinin medyanýný bulmak için dizi sýralama
	for (int i = 0; i < size; i++) {
		for (int j = i + 1; j < size; j++) {
			if (array[j] < array[i]) {
				unsigned char temp = array[i];
				array[i] = array[j];
				array[j] = temp;
			}
		}
	}

	 //Pencere piksellerinin ortanca deðerini bulma
	int median_index = size / 2;
	return array[median_index];
}

__global__ void basarisiz_median_filter(unsigned char* input_image, unsigned char* output_image, int width, int height)
{
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (col >= width || row >= height)
	{
		return;
	}

	for (int i = 0; i < WINDOW_SIZE; i++)
	{
		for (int j = 0; j < WINDOW_SIZE; j++)
		{
			int current_col = col - (WINDOW_SIZE / 2) + i;
			int current_row = row - (WINDOW_SIZE / 2) + j;

			if (current_row >= 0 && current_row < height && current_col >= 0 && current_col < width)
			{
				int index = ((current_row * width) + current_col) * 3; //pixelin 3 kanallý RGB indeksi
				if (current_col == 128 && current_row == 119)
				{
					//printf("%d / %d / %d / %d / %d \n", current_col, current_row, width, height, index);
				}
				if (i == 0 && j == 0)
				{
					output_image[(row * width + col) * 3 + R_INDEX] = input_image[index + R_INDEX];
					output_image[(row * width + col) * 3 + G_INDEX] = input_image[index + G_INDEX];
					output_image[(row * width + col) * 3 + B_INDEX] = input_image[index + B_INDEX];
				}
				else
				{
					if (input_image[index + R_INDEX] > output_image[(current_row * width + current_col) * 3 + R_INDEX])
						output_image[(current_row * width + current_col) * 3 + R_INDEX] = input_image[index + R_INDEX];
					if (input_image[index + G_INDEX] > output_image[(current_row * width + current_col) * 3 + G_INDEX])
						output_image[(current_row * width + current_col) * 3 + G_INDEX] = input_image[index + G_INDEX];
					if (input_image[index + B_INDEX] > output_image[(current_row * width + current_col) * 3 + B_INDEX])
						output_image[(current_row * width + current_col) * 3 + B_INDEX] = input_image[index + B_INDEX];
				}
			}

		}
	}


}
*/

//median filter pencere boyutu
#define WINDOW_SIZE 3

#define R_INDEX 0
#define G_INDEX 1
#define B_INDEX 2


__device__ int median_calc(int arr[], int n, int k);
__global__ void median_filter_rgb(unsigned char* input_image, unsigned char* output_image, int width, int height)
{
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (col >= 14 && col <= 17 && row >= 173 && row <= 176)
	{
		printf("col = %d row = %d input: %d/%d/%d \n", col,row,input_image[((row * width + col) * 3) + 0], input_image[((row * width + col) * 3) + 1], input_image[((row * width + col)* 3) + 2]);
	}

	if (col >= width || row >= height)
	{
		return;
	}

	int red[WINDOW_SIZE * WINDOW_SIZE];
	int green[WINDOW_SIZE * WINDOW_SIZE];
	int blue[WINDOW_SIZE * WINDOW_SIZE];

	int index = 0;

	// piksel deðerlerini topla
	for (int i = 0; i < WINDOW_SIZE; i++)
	{
		for (int j = 0; j < WINDOW_SIZE; j++)
		{

			int current_col = col - (WINDOW_SIZE / 2) + i;
			int current_row = row - (WINDOW_SIZE / 2) + j;

			if (current_row >= 0 && current_row < height && current_col >= 0 && current_col < width)
			{
				int current_index = ((current_row * width) + current_col) * 3;
				red[index] = input_image[current_index + R_INDEX];
				green[index] = input_image[current_index + G_INDEX];
				blue[index] = input_image[current_index + B_INDEX];
				if (col == 127 && row == 98)
				{
					//printf(" current_col = %d / current_row = %d \n red[%d] = %d green = %d blue = %d \n", current_col, current_row, index, red[index], green[index], blue[index]);
				}
				index++;
			}
		}
	} 
	// index 0dan 8e kadar 8de dahil ama dizide mesela red[8] maksimumdur ama fonksiyonun sonudna index++ olduðu için index 9 olur

	// piksel deðerlerini sýrala ve emdyan deðerini al

	int half_Size = WINDOW_SIZE * WINDOW_SIZE / 2;
	if (index > half_Size)
	{
		int red_median = median_calc(red, index, half_Size);
		int green_median = median_calc(green, index, half_Size);
		int blue_median = median_calc(blue, index, half_Size);

		// median deðerini çýkýþ görüntüsüne yaz

		output_image[(row * width + col) * 3 + R_INDEX] = red_median;
		output_image[(row * width + col) * 3 + G_INDEX] = green_median;
		output_image[(row * width + col) * 3 + B_INDEX] = blue_median;

		if (col >= 14 && col <= 17 && row >= 173 && row <= 176)
		{
			printf("col = % d row = % d output : %d/%d/%d\n", col, row, red_median, green_median, blue_median);
		}
	}


}


// sýralama algoritmasý medyan seçme
__device__ int median_calc(int arr[], int n, int k)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = i+1; j < n; j++)
		{
			if (arr[i] > arr[j]) 
			{
				int temp = arr[i];
				arr[i] = arr[j];
				arr[j] = temp;
			}
		}
	}
	return arr[n / 2];
} 