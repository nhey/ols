include ../Makefile

clean:
	rm -f *.out *.out.c *_opencl.py
	rm -rf __pycache__ lib

%.out: %.fut
	futhark opencl -o $@ $<

%: %.fut
	futhark pyopencl --library -o $@_opencl $<

test-fut: lib ols_test.out
	echo 2 | ./ols_test.out

test-py: lib ols_test
	python validate.py
