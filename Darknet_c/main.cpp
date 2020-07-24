#include "darknet.h"
#include <iostream>

int main()
{
	/*char cfgfile[40] = "";
	char weightfile[40] = "";
	char train_images[40] = "/data/voc/train.txt";
	char backup_directory[40] = "/home/pjreddie/backup/";*/

	char cfgfile[80] = "C:\\Users\\ASUS\\Desktop\\temp_data\\darknet\\DarkNet\\darknet\\cfg\\extraction.conv.cfg";
	char weightfile[80] = "C:\\Users\\ASUS\\Desktop\\temp_data\\data\\Yolo_v1\\extraction.conv.weights";
//	char train_images[40] = "/data/voc/train.txt";
//	char backup_directory[40] = "/home/pjreddie/backup/";

	network *net = load_network(cfgfile, weightfile, 0);
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
	return 0;
}
