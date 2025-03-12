# Here, dst file should be something like: 
# /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/llama.py
dst_file=`find / | grep packages/vllm/model_executor/models/llama.py`
cp llama.py $dst_file

# Here, dst file should be something like: 
# /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/loader.py
dst_file=`find / | grep packages/vllm/model_executor/model_loader/loader.py`
cp loader.py $dst_file

# Here, dst file should be something like: /app/vllm/benchmarks/benchmark_latency.py
# But we use an absolute path that should work in the docker
cp benchmark_latency.py /app/vllm/benchmarks
