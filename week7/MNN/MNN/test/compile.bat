rmdir /q /s build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE ..
cmake --build . --config Release
copy ..\DLL\MNN.dll .\
copy ..\DLL\opencv_world420.dll .\
copy ..\3rdparty\FreeImage3180Win32Win64\FreeImage\Dist\x64\FreeImage.dll .\