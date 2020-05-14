cmake \
-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.15 \
-DCMAKE_CXX_COMPILER:PATH=/usr/bin/g++ \
-DCMAKE_CC_COMPILER:PATH=/usr/bin/gcc \
-DPYTHON_INCLUDE_DIR:PATH=/Users/scott.daniel/AllenInstitute/miniconda3/python3.7m/ \
-DPYTHON_LIBRARY:PATH=/Users/scott.daniel/AllenInstitute/miniconda3/lib/ \
../cell-locator
