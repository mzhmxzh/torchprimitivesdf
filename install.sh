if [ -d "build" ];then
    bash clean.sh
fi

python setup.py develop
python -c "import torchprimitivesdf; print(torchprimitivesdf.__version__)"