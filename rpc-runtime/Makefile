all: rpc_dynamic.o

rpc_dynamic.o: dynamic.cpp
	${RPC_LLVM_BIN_DIR}/clang++ -c -O3 ./dynamic.cpp -o rpc_dynamic.o

clean:
	rm -f rpc_dynamic.o