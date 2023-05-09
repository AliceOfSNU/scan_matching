# scan_matching
NDT scan matching implementation in c++
![ndt overview](https://github.com/AliceOfSNU/scan_matching/assets/86138312/f8253bd0-882c-403a-aa95-28ed60b8c872)

# dependencies
pcl 1.2 or higher -> https://pointclouds.org/downloads/
eigen 3 (comes with pcl)



# build
cmake 2.8 or higher

I have built this on Ubuntu 18.04 only.
May not work on WSL or Windows due to pcl's dependencies.
to build, navigate to 'src' -> cmake . -> make
then you will see the ndt executable.

Always refer to the CMakeLists.txt if you have moved things around.

# disclaimer
some of the code is from Udacity's self driving course.
this includes 
  - helpers.h
  - helpers.cpp
  - the pcd files

the pdf, probabilities, gradient descent functions are my work.
Do not redistribute.
