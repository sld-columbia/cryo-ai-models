//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <string.h>
#include "mc_scverify.h"

#include "firmware/anomaly_detector.h"
#include "firmware/nnet_utils/nnet_helpers.h"

//hls-fpga-machine-learning weights headfiles
#include "firmware/weights/w2.h"
#include "firmware/weights/b2.h"
#include "firmware/weights/s4.h"
#include "firmware/weights/b4.h"
#include "firmware/weights/w6.h"
#include "firmware/weights/b6.h"
#include "firmware/weights/s8.h"
#include "firmware/weights/b8.h"
#include "firmware/weights/w10.h"
#include "firmware/weights/b10.h"
#include "firmware/weights/s12.h"
#include "firmware/weights/b12.h"
#include "firmware/weights/w14.h"
#include "firmware/weights/b14.h"
#include "firmware/weights/s16.h"
#include "firmware/weights/b16.h"
#include "firmware/weights/w18.h"
#include "firmware/weights/b18.h"
#include "firmware/weights/s20.h"
#include "firmware/weights/b20.h"
#include "firmware/weights/w22.h"
#include "firmware/weights/b22.h"

#define CHECKPOINT 5000

namespace nnet {
    bool trace_enabled = true;
    std::map<std::string, void *> *trace_outputs = NULL;
    size_t trace_type_size = sizeof(double);
}

template<class T> 
void print_fxd_as_bin(std::ostream &out, T data) {
    for (int j = data.length() - 1; j >= 0; j--) {
        out << data[j];
    }
}

CCS_MAIN(int argc, char **argv)
{
  //load input data from text file
  std::ifstream fin("tb_data/tb_input_features.dat");
  //load predictions from text file
  std::ifstream fpr("tb_data/tb_output_predictions.dat");
  
  //hls-fpga-machine-learning insert load weights
  nnet::load_weights_from_txt<weight2_t, 8192>(w2, "w2.txt");
  nnet::load_weights_from_txt<bias2_t, 64>(b2, "b2.txt");
  nnet::load_weights_from_txt<batch_normalization_scale_t, 64>(s4, "s4.txt");
  nnet::load_weights_from_txt<batch_normalization_bias_t, 64>(b4, "b4.txt");
  nnet::load_weights_from_txt<weight6_t, 4096>(w6, "w6.txt");
  nnet::load_weights_from_txt<bias6_t, 64>(b6, "b6.txt");
  nnet::load_weights_from_txt<batch_normalization_1_scale_t, 64>(s8, "s8.txt");
  nnet::load_weights_from_txt<batch_normalization_1_bias_t, 64>(b8, "b8.txt");
  nnet::load_weights_from_txt<weight10_t, 512>(w10, "w10.txt");
  nnet::load_weights_from_txt<bias10_t, 8>(b10, "b10.txt");
  nnet::load_weights_from_txt<batch_normalization_2_scale_t, 8>(s12, "s12.txt");
  nnet::load_weights_from_txt<batch_normalization_2_bias_t, 8>(b12, "b12.txt");
  nnet::load_weights_from_txt<weight14_t, 512>(w14, "w14.txt");
  nnet::load_weights_from_txt<bias14_t, 64>(b14, "b14.txt");
  nnet::load_weights_from_txt<batch_normalization_3_scale_t, 64>(s16, "s16.txt");
  nnet::load_weights_from_txt<batch_normalization_3_bias_t, 64>(b16, "b16.txt");
  nnet::load_weights_from_txt<weight18_t, 4096>(w18, "w18.txt");
  nnet::load_weights_from_txt<bias18_t, 64>(b18, "b18.txt");
  nnet::load_weights_from_txt<batch_normalization_4_scale_t, 64>(s20, "s20.txt");
  nnet::load_weights_from_txt<batch_normalization_4_bias_t, 64>(b20, "b20.txt");
  nnet::load_weights_from_txt<weight22_t, 8192>(w22, "w22.txt");
  nnet::load_weights_from_txt<bias22_t, 128>(b22, "b22.txt");

	std::cout << "Mentor Graphics Catapult HLS" << std::endl;
#ifdef RTL_SIM
  std::string RESULTS_LOG = "tb_data/catapult_rtl_cosim_results.log";
#else
  std::string RESULTS_LOG = "tb_data/catapult_csim_results.log";
#endif
  std::ofstream fout(RESULTS_LOG);

	std::string INPUT_FILE_BIN_MEM = "tb_data/tb_input_features.mem";
  std::string OUTPUT_FILE_BIN_MEM = "tb_data/tb_output_predictions.mem";
  std::ofstream fout_ifbm(INPUT_FILE_BIN_MEM);
  std::ofstream fout_ofbm(OUTPUT_FILE_BIN_MEM);

  //save traces step1
  std::string W2_FILE_BIN_MEM = "tb_data/w2.mem";
  std::ofstream fout_w2fbm(W2_FILE_BIN_MEM);
  std::string B2_FILE_BIN_MEM = "tb_data/b2.mem";
  std::ofstream fout_b2fbm(B2_FILE_BIN_MEM);
  std::string S4_FILE_BIN_MEM = "tb_data/s4.mem";
  std::ofstream fout_s4fbm(S4_FILE_BIN_MEM);
  std::string B4_FILE_BIN_MEM = "tb_data/b4.mem";
  std::ofstream fout_b4fbm(B4_FILE_BIN_MEM);
  std::string W6_FILE_BIN_MEM = "tb_data/w6.mem";
  std::ofstream fout_w6fbm(W6_FILE_BIN_MEM);
  std::string B6_FILE_BIN_MEM = "tb_data/b6.mem";
  std::ofstream fout_b6fbm(B6_FILE_BIN_MEM);
  std::string S8_FILE_BIN_MEM = "tb_data/s8.mem";
  std::ofstream fout_s8fbm(S8_FILE_BIN_MEM);
  std::string B8_FILE_BIN_MEM = "tb_data/b8.mem";
  std::ofstream fout_b8fbm(B8_FILE_BIN_MEM);
  std::string W10_FILE_BIN_MEM = "tb_data/w10.mem";
  std::ofstream fout_w10fbm(W10_FILE_BIN_MEM);
  std::string B10_FILE_BIN_MEM = "tb_data/b10.mem";
  std::ofstream fout_b10fbm(B10_FILE_BIN_MEM);
  std::string S12_FILE_BIN_MEM = "tb_data/s12.mem";
  std::ofstream fout_s12fbm(S12_FILE_BIN_MEM);
  std::string B12_FILE_BIN_MEM = "tb_data/b12.mem";
  std::ofstream fout_b12fbm(B12_FILE_BIN_MEM);
  std::string W14_FILE_BIN_MEM = "tb_data/w14.mem";
  std::ofstream fout_w14fbm(W14_FILE_BIN_MEM);
  std::string B14_FILE_BIN_MEM = "tb_data/b14.mem";
  std::ofstream fout_b14fbm(B14_FILE_BIN_MEM);
  std::string S16_FILE_BIN_MEM = "tb_data/s16.mem";
  std::ofstream fout_s16fbm(S16_FILE_BIN_MEM);
  std::string B16_FILE_BIN_MEM = "tb_data/b16.mem";
  std::ofstream fout_b16fbm(B16_FILE_BIN_MEM);
  std::string W18_FILE_BIN_MEM = "tb_data/w18.mem";
  std::ofstream fout_w18fbm(W18_FILE_BIN_MEM);
  std::string B18_FILE_BIN_MEM = "tb_data/b18.mem";
  std::ofstream fout_b18fbm(B18_FILE_BIN_MEM);
  std::string S20_FILE_BIN_MEM = "tb_data/s20.mem";
  std::ofstream fout_s20fbm(S20_FILE_BIN_MEM);
  std::string B20_FILE_BIN_MEM = "tb_data/b20.mem";
  std::ofstream fout_b20fbm(B20_FILE_BIN_MEM);
  std::string W22_FILE_BIN_MEM = "tb_data/w22.mem";
  std::ofstream fout_w22fbm(W22_FILE_BIN_MEM);
  std::string B22_FILE_BIN_MEM = "tb_data/b22.mem";
  std::ofstream fout_b22fbm(B22_FILE_BIN_MEM);

  std::string iline;
  std::string pline;
  int e = 0;

  if (fin.is_open() && fpr.is_open()) {
    while ( std::getline(fin,iline) && std::getline (fpr,pline) ) {
      if (e % CHECKPOINT == 0) std::cout << "Processing input " << e << std::endl;
      e++;
      char* cstr=const_cast<char*>(iline.c_str());
      char* current;
      std::vector<float> in;
      current=strtok(cstr," ");
      while(current!=NULL) {
        in.push_back(atof(current));
        current=strtok(NULL," ");
      }
      cstr=const_cast<char*>(pline.c_str());
      std::vector<float> pr;
      current=strtok(cstr," ");
      while(current!=NULL) {
        pr.push_back(atof(current));
        current=strtok(NULL," ");
      }

      //hls-fpga-machine-learning insert data
      std::vector<float>::const_iterator in_begin = in.cbegin();
      std::vector<float>::const_iterator in_end;
      input_t input_1[N_INPUT_1_1];
      in_end = in_begin + (N_INPUT_1_1);
      std::copy(in_begin, in_end, input_1);
      in_begin = in_end;
      layer22_t layer22_out[N_LAYER_22]{};
      std::fill_n(layer22_out, 128, 0.);

      //hls-fpga-machine-learning insert top-level-function
      unsigned short size_in1,size_out1;
      CCS_DESIGN(anomaly_detector)(input_1,layer22_out,size_in1,size_out1,w2,b2,s4,b4,w6,b6,s8,b8,w10,b10,s12,b12,w14,b14,s16,b16,w18,b18,s20,b20,w22,b22);

      //hls-fpga-machine-learning insert tb-output
      for(int i = 0; i < N_LAYER_22; i++) {
        fout << layer22_out[i] << " ";
      }
      fout << std::endl;

      if (e % CHECKPOINT == 0) {
        std::cout << "Predictions" << std::endl;
        //hls-fpga-machine-learning insert predictions
        for(int i = 0; i < N_LAYER_22; i++) {
          std::cout << pr[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Quantized predictions" << std::endl;
        //hls-fpga-machine-learning insert quantized
        for(int i = 0; i < N_LAYER_22; i++) {
          std::cout << layer22_out[i] << " ";
        }
        std::cout << std::endl;
      }
      
      //save traces step2
      /*for(int i = size_in1-1; i >=0; i--) {
        print_fxd_as_bin<input_t>(fout_ifbm, input_1[i]);
      }
      fout_ifbm << std::endl;
      for(int i = N_LAYER_22-1; i >=0; i--) {
        print_fxd_as_bin<result_t>(fout_ofbm, layer22_out[i]);
      }
      fout_ofbm << std::endl;
      static bool one_time = true;
      if (one_time) {
      unsigned int size_w2 = 8192;
      for(int i = size_w2-1; i >= 0; i--) {
        print_fxd_as_bin<weight2_t>(fout_w2fbm, w2[i]);
      }
      fout_w2fbm << std::endl;
      unsigned int size_b2 = 64;
      for(int i = size_b2-1; i >= 0; i--) {
        print_fxd_as_bin<bias2_t>(fout_b2fbm, b2[i]);
      }
      fout_b2fbm << std::endl;
      unsigned int size_s4 = 64;
      for(int i = size_s4-1; i >= 0; i--) {
        print_fxd_as_bin<batch_normalization_scale_t>(fout_s4fbm, s4[i]);
      }
      fout_s4fbm << std::endl;
      unsigned int size_b4 = 64;
      for(int i = size_b4-1; i >= 0; i--) {
        print_fxd_as_bin<batch_normalization_bias_t>(fout_b4fbm, b4[i]);
      }
      fout_b4fbm << std::endl;
      unsigned int size_w6 = 4096;
      for(int i = size_w6-1; i >= 0; i--) {
        print_fxd_as_bin<weight6_t>(fout_w6fbm, w6[i]);
      }
      fout_w6fbm << std::endl;
      unsigned int size_b6 = 64;
      for(int i = size_b6-1; i >= 0; i--) {
        print_fxd_as_bin<bias6_t>(fout_b6fbm, b6[i]);
      }
      fout_b6fbm << std::endl;
      unsigned int size_s8 = 64;
      for(int i = size_s8-1; i >= 0; i--) {
        print_fxd_as_bin<batch_normalization_1_scale_t>(fout_s8fbm, s8[i]);
      }
      fout_s8fbm << std::endl;
      unsigned int size_b8 = 64;
      for(int i = size_b8-1; i >= 0; i--) {
        print_fxd_as_bin<batch_normalization_1_bias_t>(fout_b8fbm, b8[i]);
      }
      fout_b8fbm << std::endl;
      unsigned int size_w10 = 512;
      for(int i = size_w10-1; i >= 0; i--) {
        print_fxd_as_bin<weight10_t>(fout_w10fbm, w10[i]);
      }
      fout_w10fbm << std::endl;
      unsigned int size_b10 = 8;
      for(int i = size_b10-1; i >= 0; i--) {
        print_fxd_as_bin<bias10_t>(fout_b10fbm, b10[i]);
      }
      fout_b10fbm << std::endl;
      unsigned int size_s12 = 8;
      for(int i = size_s12-1; i >= 0; i--) {
        print_fxd_as_bin<batch_normalization_2_scale_t>(fout_s12fbm, s12[i]);
      }
      fout_s12fbm << std::endl;
      unsigned int size_b12 = 8;
      for(int i = size_b12-1; i >= 0; i--) {
        print_fxd_as_bin<batch_normalization_2_bias_t>(fout_b12fbm, b12[i]);
      }
      fout_b12fbm << std::endl;
      unsigned int size_w14 = 512;
      for(int i = size_w14-1; i >= 0; i--) {
        print_fxd_as_bin<weight14_t>(fout_w14fbm, w14[i]);
      }
      fout_w14fbm << std::endl;
      unsigned int size_b14 = 64;
      for(int i = size_b14-1; i >= 0; i--) {
        print_fxd_as_bin<bias14_t>(fout_b14fbm, b14[i]);
      }
      fout_b14fbm << std::endl;
      unsigned int size_s16 = 64;
      for(int i = size_s16-1; i >= 0; i--) {
        print_fxd_as_bin<batch_normalization_3_scale_t>(fout_s16fbm, s16[i]);
      }
      fout_s16fbm << std::endl;
      unsigned int size_b16 = 64;
      for(int i = size_b16-1; i >= 0; i--) {
        print_fxd_as_bin<batch_normalization_3_bias_t>(fout_b16fbm, b16[i]);
      }
      fout_b16fbm << std::endl;
      unsigned int size_w18 = 4096;
      for(int i = size_w18-1; i >= 0; i--) {
        print_fxd_as_bin<weight18_t>(fout_w18fbm, w18[i]);
      }
      fout_w18fbm << std::endl;
      unsigned int size_b18 = 64;
      for(int i = size_b18-1; i >= 0; i--) {
        print_fxd_as_bin<bias18_t>(fout_b18fbm, b18[i]);
      }
      fout_b18fbm << std::endl;
      unsigned int size_s20 = 64;
      for(int i = size_s20-1; i >= 0; i--) {
        print_fxd_as_bin<batch_normalization_4_scale_t>(fout_s20fbm, s20[i]);
      }
      fout_s20fbm << std::endl;
      unsigned int size_b20 = 64;
      for(int i = size_b20-1; i >= 0; i--) {
        print_fxd_as_bin<batch_normalization_4_bias_t>(fout_b20fbm, b20[i]);
      }
      fout_b20fbm << std::endl;
      unsigned int size_w22 = 8192;
      for(int i = size_w22-1; i >= 0; i--) {
        print_fxd_as_bin<weight22_t>(fout_w22fbm, w22[i]);
      }
      fout_w22fbm << std::endl;
      unsigned int size_b22 = 128;
      for(int i = size_b22-1; i >= 0; i--) {
        print_fxd_as_bin<bias22_t>(fout_b22fbm, b22[i]);
      }
      fout_b22fbm << std::endl;
      one_time = false;
      }*/
    }
    fin.close();
    fpr.close();
  } else {
    std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;
    //hls-fpga-machine-learning insert zero
    input_t input_1[N_INPUT_1_1];
    std::fill_n(input_1, N_INPUT_1_1, 0.);
    layer22_t layer22_out[N_LAYER_22]{};
      std::fill_n(layer22_out, 128, 0.);

    //hls-fpga-machine-learning insert top-level-function
    unsigned short size_in1,size_out1;
    CCS_DESIGN(anomaly_detector)(input_1,layer22_out,size_in1,size_out1,w2,b2,s4,b4,w6,b6,s8,b8,w10,b10,s12,b12,w14,b14,s16,b16,w18,b18,s20,b20,w22,b22);

    //hls-fpga-machine-learning insert output
    for(int i = 0; i < N_LAYER_22; i++) {
      std::cout << layer22_out[i] << " ";
    }
    std::cout << std::endl;

    //save traces step2
    /*for(int i = size_in1-1; i >=0; i--) {
      print_fxd_as_bin<input_t>(fout_ifbm, input_1[i]);
    }
    fout_ifbm << std::endl;
    for(int i = N_LAYER_22-1; i >=0; i--) {
      print_fxd_as_bin<result_t>(fout_ofbm, layer22_out[i]);
    }
    fout_ofbm << std::endl;
    static bool one_time = true;
    if (one_time) {
    unsigned int size_w2 = 8192;
    for(int i = size_w2-1; i >= 0; i--) {
      print_fxd_as_bin<weight2_t>(fout_w2fbm, w2[i]);
    }
    fout_w2fbm << std::endl;
    unsigned int size_b2 = 64;
    for(int i = size_b2-1; i >= 0; i--) {
      print_fxd_as_bin<bias2_t>(fout_b2fbm, b2[i]);
    }
    fout_b2fbm << std::endl;
    unsigned int size_s4 = 64;
    for(int i = size_s4-1; i >= 0; i--) {
      print_fxd_as_bin<batch_normalization_scale_t>(fout_s4fbm, s4[i]);
    }
    fout_s4fbm << std::endl;
    unsigned int size_b4 = 64;
    for(int i = size_b4-1; i >= 0; i--) {
      print_fxd_as_bin<batch_normalization_bias_t>(fout_b4fbm, b4[i]);
    }
    fout_b4fbm << std::endl;
    unsigned int size_w6 = 4096;
    for(int i = size_w6-1; i >= 0; i--) {
      print_fxd_as_bin<weight6_t>(fout_w6fbm, w6[i]);
    }
    fout_w6fbm << std::endl;
    unsigned int size_b6 = 64;
    for(int i = size_b6-1; i >= 0; i--) {
      print_fxd_as_bin<bias6_t>(fout_b6fbm, b6[i]);
    }
    fout_b6fbm << std::endl;
    unsigned int size_s8 = 64;
    for(int i = size_s8-1; i >= 0; i--) {
      print_fxd_as_bin<batch_normalization_1_scale_t>(fout_s8fbm, s8[i]);
    }
    fout_s8fbm << std::endl;
    unsigned int size_b8 = 64;
    for(int i = size_b8-1; i >= 0; i--) {
      print_fxd_as_bin<batch_normalization_1_bias_t>(fout_b8fbm, b8[i]);
    }
    fout_b8fbm << std::endl;
    unsigned int size_w10 = 512;
    for(int i = size_w10-1; i >= 0; i--) {
      print_fxd_as_bin<weight10_t>(fout_w10fbm, w10[i]);
    }
    fout_w10fbm << std::endl;
    unsigned int size_b10 = 8;
    for(int i = size_b10-1; i >= 0; i--) {
      print_fxd_as_bin<bias10_t>(fout_b10fbm, b10[i]);
    }
    fout_b10fbm << std::endl;
    unsigned int size_s12 = 8;
    for(int i = size_s12-1; i >= 0; i--) {
      print_fxd_as_bin<batch_normalization_2_scale_t>(fout_s12fbm, s12[i]);
    }
    fout_s12fbm << std::endl;
    unsigned int size_b12 = 8;
    for(int i = size_b12-1; i >= 0; i--) {
      print_fxd_as_bin<batch_normalization_2_bias_t>(fout_b12fbm, b12[i]);
    }
    fout_b12fbm << std::endl;
    unsigned int size_w14 = 512;
    for(int i = size_w14-1; i >= 0; i--) {
      print_fxd_as_bin<weight14_t>(fout_w14fbm, w14[i]);
    }
    fout_w14fbm << std::endl;
    unsigned int size_b14 = 64;
    for(int i = size_b14-1; i >= 0; i--) {
      print_fxd_as_bin<bias14_t>(fout_b14fbm, b14[i]);
    }
    fout_b14fbm << std::endl;
    unsigned int size_s16 = 64;
    for(int i = size_s16-1; i >= 0; i--) {
      print_fxd_as_bin<batch_normalization_3_scale_t>(fout_s16fbm, s16[i]);
    }
    fout_s16fbm << std::endl;
    unsigned int size_b16 = 64;
    for(int i = size_b16-1; i >= 0; i--) {
      print_fxd_as_bin<batch_normalization_3_bias_t>(fout_b16fbm, b16[i]);
    }
    fout_b16fbm << std::endl;
    unsigned int size_w18 = 4096;
    for(int i = size_w18-1; i >= 0; i--) {
      print_fxd_as_bin<weight18_t>(fout_w18fbm, w18[i]);
    }
    fout_w18fbm << std::endl;
    unsigned int size_b18 = 64;
    for(int i = size_b18-1; i >= 0; i--) {
      print_fxd_as_bin<bias18_t>(fout_b18fbm, b18[i]);
    }
    fout_b18fbm << std::endl;
    unsigned int size_s20 = 64;
    for(int i = size_s20-1; i >= 0; i--) {
      print_fxd_as_bin<batch_normalization_4_scale_t>(fout_s20fbm, s20[i]);
    }
    fout_s20fbm << std::endl;
    unsigned int size_b20 = 64;
    for(int i = size_b20-1; i >= 0; i--) {
      print_fxd_as_bin<batch_normalization_4_bias_t>(fout_b20fbm, b20[i]);
    }
    fout_b20fbm << std::endl;
    unsigned int size_w22 = 8192;
    for(int i = size_w22-1; i >= 0; i--) {
      print_fxd_as_bin<weight22_t>(fout_w22fbm, w22[i]);
    }
    fout_w22fbm << std::endl;
    unsigned int size_b22 = 128;
    for(int i = size_b22-1; i >= 0; i--) {
      print_fxd_as_bin<bias22_t>(fout_b22fbm, b22[i]);
    }
    fout_b22fbm << std::endl;
    one_time = false;
    }*/

    //hls-fpga-machine-learning insert tb-output
    for(int i = 0; i < N_LAYER_22; i++) {
      fout << layer22_out[i] << " ";
    }
    fout << std::endl;
  }

  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

	fout_ifbm.close();
  std::cout << "INFO: Saved input data to .mem file: " << INPUT_FILE_BIN_MEM << std::endl;
  fout_ofbm.close();
  std::cout << "INFO: Saved output data to .mem file: " << OUTPUT_FILE_BIN_MEM << std::endl;
  
  //save traces step3
  fout_w2fbm.close();
  fout_b2fbm.close();
  fout_s4fbm.close();
  fout_b4fbm.close();
  fout_w6fbm.close();
  fout_b6fbm.close();
  fout_s8fbm.close();
  fout_b8fbm.close();
  fout_w10fbm.close();
  fout_b10fbm.close();
  fout_s12fbm.close();
  fout_b12fbm.close();
  fout_w14fbm.close();
  fout_b14fbm.close();
  fout_s16fbm.close();
  fout_b16fbm.close();
  fout_w18fbm.close();
  fout_b18fbm.close();
  fout_s20fbm.close();
  fout_b20fbm.close();
  fout_w22fbm.close();
  fout_b22fbm.close();

  CCS_RETURN(0);
}
