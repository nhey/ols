# May want to run `futhark pkg sync` in ../.
clean:
	rm -f *.out *.out.c *_pyopencl.py
	rm -rf __pycache__

%.out: %.fut
	futhark opencl -o $@ $<

%_pyopencl: %.fut
	futhark pyopencl --library -o $@ $<

test-fut:
	futhark test ols_test.fut
	rm ols_test.c
	rm ols_test

test-py: ols_test_pyopencl
	python validate.py
