help:
	@echo "INFO: make <TAB> for targets"
.PHONY: help

syn-gui:
	catapult -product ultra -f build_prj.tcl
.PHONY: syn-gui

clean:
	rm -rf Catapult*
	rm -f catapult.log
	rm -f design_checker_constraints.tcl
	rm -f design_checker_pre_build.tcl
	rm -f tb_data/*.mem
	rm -f tb_data/*.log
.PHONY: clean

ultraclean: clean
	rm -rf catapult_cache_*
.PHONY: ultraclean
