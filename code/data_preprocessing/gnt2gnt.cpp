#include "stdafx.h"

int num = 0;

void grayNorm(unsigned char buff[], unsigned char new_buff[])
{
	short rows, cols;
	/*char test[3] = "\0";
	memcpy(test, buff + 4, 2);
	cout << test << endl;*/
	
	memcpy(&cols, buff + 6, 2);
	memcpy(&rows, buff + 8, 2);
	
	Mat cut(rows, cols, CV_8UC1, buff + 10);
	////////////////////灰度增强/////////////////////
	bitwise_not(cut, cut);

	float meanValue = 0.0f;
	float stdValue = 0.0f;
	int cnt = 0;
	for (size_t i = 0; i < cut.cols; i++)
	{
		for (size_t j = 0; j < cut.rows; j++)
			if (cut.at<uchar>(j, i) > 0)
			{
				cnt++;
				meanValue += cut.at<uchar>(j, i);
			}

	}
	meanValue /= (cnt + 0.1f);

	if (meanValue < 110)
	{
		num++;	
		for (size_t i = 0; i < cut.cols; i++)
		{
			for (size_t j = 0; j < cut.rows; j++)
				if (cut.at<uchar>(j, i) > 0)
				{
					stdValue += (cut.at<uchar>(j, i) - meanValue)*(cut.at<uchar>(j, i) - meanValue);
				}
		}

		GaussianBlur(cut, cut, Size(3, 3), 0, 0);
		stdValue /= (cnt + 0.1f);
		stdValue = sqrt(stdValue);

		float power = log(185.0f / (185.0f + 40.0f)) / log(meanValue / (meanValue + 2 * stdValue));// method from Prof Liu
		float alpha = 185.0f / pow(meanValue, power);// method from Prof Liu
		for (size_t i = 0; i < cut.cols; i++)
		{
			for (size_t j = 0; j < cut.rows; j++)
			{
				int grayValue = cut.at<uchar>(j, i);
				if (grayValue > 0)
				{
					cut.at<uchar>(j, i) = min(255.0, alpha*pow(grayValue, power));// method from Prof Liu
				}
			}
		}
	}
	//bitwise_not(cut, cut);
	Mat cut_resize;
	resize(cut, cut_resize, Size(112, 112));
	//
	//cvtColor(cut, rgb, COLOR_GRAY2BGR);
	//imwrite("K:/test1.jpg", rgb);
	
	/*for (int i = 0; i < 112; i++) {
		for (int j = 0; j < 112; j++) {
			for (int k = 0; k < 3; k++) {
				new_buff[i * 112 * 3 + j * 3 + k] = cut_resize.at<uchar>(i, j);
			}			
		}
	}	*/
	for (int i = 0; i < 112; i++) {
		for (int j = 0; j < 112; j++) {
			new_buff[i * 112 + j] = cut_resize.at<uchar>(i, j);
		}
	}
	cut_resize.release();
	cut.release();
}
int main(int argc, char* argv[])
{
	FILE *fpIn, *fpOut;	
	char fname[100] = "K:/华为生态数据GNT格式/hwtst.gnt";
	fopen_s(&fpIn, fname, "rb");
	if (fpIn == 0)
	{
		printf("Cannot open the file %s\n", fname);
		exit(1);
	}

	long gntlen;
	char label[10];
	unsigned char* buff;
	unsigned char* new_buff;
	char fname_output[100] = "K:/华为生态数据GNT格式/hwtst-graynorm.gnt";
	printf("Output file: %s\n", fname_output);
	fopen_s(&fpOut, fname_output, "wb");

	_int64 fpLen;
	_fseeki64(fpIn, 0, SEEK_END);
	fpLen = _ftelli64(fpIn);
	_fseeki64(fpIn, 0, SEEK_SET);

	//int temp_count = 0;
	while (_ftelli64(fpIn)<fpLen)
	{
		fread(&gntlen, 4, 1, fpIn);
		buff = new unsigned char[gntlen];
		//int new_gntlen = 112 * 112 * 3 + 10;
		int new_gntlen = 112 * 112 * 1 + 10;
		short new_height = 112;
		new_buff = new unsigned char[new_gntlen - 10];
		_fseeki64(fpIn, -4, SEEK_CUR);    
		fwrite(&new_gntlen, sizeof(int), 1, fpOut);
		fread(buff, gntlen, 1, fpIn);
		fwrite(buff + 4, 2, 1, fpOut);//label
		/*char test[3] = "\0";
		memcpy(test, buff + 4, 2);
		cout << test << endl;*/

		grayNorm(buff, new_buff);
		
		fwrite(&new_height, 2, 1, fpOut);
		fwrite(&new_height, 2, 1, fpOut);
		fwrite(new_buff, new_gntlen - 10, 1, fpOut);

		/*Mat cut;
		cut = Mat::zeros(112, 112, CV_8UC3);		
		cut.data = new_buff;
		imwrite("K:/test.jpg", cut);
		cut.release();
		temp_count++;
		if(temp_count==10)
			break;*/

		delete new_buff;
		delete buff;
	}
	printf("Transformed: %d\n", num);

	fclose(fpIn);
	fclose(fpOut);
	system("Pause");
	return 0;
}
