# Uninstall 
pip uninstall arc -y
rm -rf /opt/conda/lib/python3.11/site-packages/hf_rocm_kernels/
rm -rf build

# Install
pip install --upgrade pip

pip install --upgrade numba scipy huggingface-hub[cli]
pip install "numpy<2"
pip install -r requirements-rocm.txt
pip install setuptools_scm
pip install matplotlib
pip install tabulate

export PYTORCH_ROCM_ARCH=$1
python3 setup.py develop
