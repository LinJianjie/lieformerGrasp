### install the setup
    singularity exec --nv ~/pytorch_env_1_7.sif python3 setup.py install --prefix=$PWD/../../../extensions_build/chamfer_distance
### add the path in the code
    import sys
    sys.path.append("/home/lin/cd/lib/python3.6/site-packages/chamfer-2.0.0-py3.6-linux-x86_64.egg")