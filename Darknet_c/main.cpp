#include "darknet.h"
#include <iostream>

char *voc_names[] = { (char*)"aeroplane", (char*)"bicycle", (char*)"bird", (char*)"boat",  (char*)"bottle", 
(char*)"bus", (char*)"car", (char*)"cat", (char*)"chair", (char*)"cow", 
(char*)"diningtable", (char*)"dog", (char*)"horse", (char*)"motorbike",(char*)"person", 
(char*)"pottedplant", (char*)"sheep", (char*)"sofa", (char*)"train", (char*)"tvmonitor" };

int main()
{
	/*char cfgfile[40] = "";
	char weightfile[40] = "";
	char train_images[40] = "/data/voc/train.txt";
	char backup_directory[40] = "/home/pjreddie/backup/";*/

	//char cfgfile[80] = "C:\\Users\\ASUS\\Desktop\\temp_data\\darknet\\DarkNet\\darknet\\cfg\\extraction.conv.cfg";
	//char weightfile[80] = "C:\\Users\\ASUS\\Desktop\\temp_data\\data\\Yolo_v1\\extraction.conv.weights";
	char cfgfile[80] = "C:\\Users\\ASUS\\Desktop\\temp_data\\darknet\\DarkNet\\darknet\\cfg\\yolov1.cfg";
	char weightfile[80] = "C:\\Users\\ASUS\\Desktop\\temp_data\\data\\Yolo_v1\\pre-trained\\yolov1.weights";
	char filename[100] = "C:\\Users\\ASUS\\Desktop\\temp_data\\darknet\\DarkNet\\darknet\\data\\test_4.jpg";
	float thresh = 0.10;
	//	char train_images[40] = "/data/voc/train.txt";
	//	char backup_directory[40] = "/home/pjreddie/backup/";

	image **alphabet = load_alphabet();
	network *net = load_network(cfgfile, weightfile, 0);
	layer l = net->layers[net->n - 1];
	set_batch_network(net, 1);
	srand(2222222);
	clock_t time;
	char buff[256];
	char *input = buff;
	float nms = .4;
	while (1) {
		if (filename) {
			strncpy(input, filename, 256);
		}
		else {
			printf("Enter Image Path: ");
			fflush(stdout);
			input = fgets(input, 256, stdin);
			if (!input) return 0;
			strtok(input, "\n");
		}
		image im = load_image_color(input, 0, 0);
		image sized = resize_image(im, net->w, net->h);
		float *X = sized.data;
		time = clock();
		network_predict(net, X);
		printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));

		int nboxes = 0;
		detection *dets = get_network_boxes(net, 1, 1, thresh, 0, 0, 0, &nboxes);
		if (nms) do_nms_sort(dets, l.side*l.side*l.n, l.classes, nms);

		draw_detections(im, dets, l.side*l.side*l.n, thresh, voc_names, alphabet, 20);
		save_image(im, "predictions");
		show_image(im, "predictions", 0);
		free_detections(dets, nboxes);
		free_image(im);
		free_image(sized);
		if (filename) break;
	}
	return 0;
}
