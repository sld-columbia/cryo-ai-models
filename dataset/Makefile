download: \
	downsampled_128_5_to_32_4_ground_truths_skip_method.npy \
	downsampled_128_5_to_32_4_skip_method.npy \
	ad03_tb_input_features.dat \
	ad03_tb_output_predictions.dat
.PHONY: download

downsampled_128_5_to_32_4_ground_truths_skip_method.npy:
	wget https://www.dropbox.com/s/z2m3vn1siqku0bu/downsampled_128_5_to_32_4_ground_truths_skip_method.npy?dl=0 -O downsampled_128_5_to_32_4_ground_truths_skip_method.npy

downsampled_128_5_to_32_4_skip_method.npy:
	wget https://www.dropbox.com/s/ht98vzmi7l6t5hr/downsampled_128_5_to_32_4_skip_method.npy?dl=0 -O downsampled_128_5_to_32_4_skip_method.npy

ad03_tb_input_features.dat:
	wget https://www.dropbox.com/s/rbtxywa07zyxcid/ad03_tb_input_features.dat?dl=0 -O ad03_tb_input_features.dat

ad03_tb_output_predictions.dat:
	wget https://www.dropbox.com/s/9mxektsivavk6fg/ad03_tb_output_predictions.dat?dl=0 -O ad03_tb_output_predictions.dat

clean:
	rm -f *.dat *.npy
.PHONY: clean
