set_working_dir .
solution new -state initial
solution options defaults
solution options set Flows/ModelSim/VLOG_OPTS {-suppress 12110}
solution options set Flows/ModelSim/VSIM_OPTS {-t ps -suppress 12110}
solution options set Flows/DesignCompiler/OutNetlistFormat verilog
solution options set /Input/CppStandard c++11
set CATAPULT_VERSION  [string map { / - } [string map { . - } [application get /SYSTEM/RELEASE_VERSION]]]
solution options set Cache/UserCacheHome "catapult_cache_$CATAPULT_VERSION"
solution options set Cache/DefaultCacheHomeEnabled false
flow package require /SCVerify
directive set -REGISTER_THRESHOLD 25600
solution options set Flows/QuestaSIM/SCCOM_OPTS {-g -x c++ -Wall -Wno-unused-label -Wno-unknown-pragmas -DRTL_SIM -D__ASIC__}
solution options set /Input/CompilerFlags {-DMNTR_CATAPULT_HLS -D__ASIC__}
solution options set /Input/SearchPath {../anomaly_detector_prj/firmware/ ../anomaly_detector_prj/firmware/ap_types ../anomaly_detector_prj/firmware/weights ../anomaly_detector_prj/firmware/nnet_utils}
solution file add ../anomaly_detector_prj/firmware/anomaly_detector.cpp -type C++
solution file add ../anomaly_detector_prj/anomaly_detector_test.cpp -type C++ -exclude true
solution file set ../anomaly_detector_prj/anomaly_detector_test.cpp -args {-D__WEIGHTS_FROM_FILE__ -D__WEIGHTS_DIR__=../OutputDir/firmware/weights/}
solution option set Output/OutputVHDL false
solution option set Output/OutputVerilog true
solution option set Output/PackageOutput true
solution option set Output/PackageStaticFiles true
solution option set Output/PrefixStaticFiles true
#solution options set Output/SubBlockNamePrefix "encoder_"
solution option set Output/DoNotModifyNames true
go new
go analyze
directive set -DESIGN_HIERARCHY anomaly_detector
solution library add nangate-45nm_beh -- -rtlsyntool RTLCompiler -vendor Nangate -technology 045nm
solution library add ccs_sample_mem
go libraries
directive set -CLOCKS { \
        clk { \
            -CLOCK_PERIOD 10 \
            -CLOCK_EDGE rising \
            -CLOCK_HIGH_TIME 12.5 \
            -CLOCK_OFFSET 0.000000 \
            -CLOCK_UNCERTAINTY 0.0 \
            -RESET_KIND sync \
            -RESET_SYNC_NAME rst \
            -RESET_SYNC_ACTIVE high \
            -RESET_ASYNC_NAME arst_n \
            -RESET_ASYNC_ACTIVE low \
            -ENABLE_NAME {} \
            -ENABLE_ACTIVE high \
        } \
    }
directive set /anomaly_detector -GATE_EFFORT normal
go assembly


directive set /anomaly_detector/input_1:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/input_1:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/layer22_out:rsc -MAP_TO_MODULE ccs_ioport.ccs_out
directive set /anomaly_detector/layer22_out:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/const_size_in_1:rsc -MAP_TO_MODULE ccs_ioport.ccs_out
directive set /anomaly_detector/const_size_out_1:rsc -MAP_TO_MODULE ccs_ioport.ccs_out

#set directives for all of the inputs ports
directive set /anomaly_detector/w2:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/w2:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/b2:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/b2:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/w6:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/w6:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/b6:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/b6:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/w10:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/w10:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/b10:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/b10:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/w14:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/w14:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/b14:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/b14:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/w18:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/w18:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/b18:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/b18:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/w22:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/w22:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/b22:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/b22:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/s4:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/s4:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/b4:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/b4:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/s8:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/s8:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/b8:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/b8:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/s12:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/s12:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/b12:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/b12:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/s16:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/s16:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/b16:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/b16:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/s20:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/s20:rsc -PACKING_MODE sidebyside
directive set /anomaly_detector/b20:rsc -MAP_TO_MODULE ccs_ioport.ccs_in
directive set /anomaly_detector/b20:rsc -PACKING_MODE sidebyside

#directive set /anomaly_detector/core -MAX_LATENCY 8
directive set /anomaly_detector/core -DESIGN_GOAL Area
flow run /SCVerify/launch_make ./scverify/Verify_orig_cxx_osci.mk {} SIMTOOL=osci sim
go architect
go allocate
go extract
