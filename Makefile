############################
#  	    * PROFILING *      #
############################


profile = RUSTFLAGS='-g' cargo build --release --bin $(1); \
		valgrind --tool=callgrind --callgrind-out-file=callgrind.out \
		--collect-jumps=yes --simulate-cache=yes \
		./target/release/$(1)


profile.test_MP2_par: 
	$(call profile,SCF_MRJD_RS)
