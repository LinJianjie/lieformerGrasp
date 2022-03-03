# Point Transformer with Lie Algebra Grasp Representation for an End-to-End 6-DOF Grasp Detection

### Usage
#### 1) Prerequisite
1. Install dependencies via `pip3 install -r requirments.txt`.
2. Follow [this guide](http://open3d.org/docs/getting_started.html) to install Open3D for point cloud I/O.
3. Build the point cloud distance metric in the folder 
   ```transformergrap\pytorch_utils\components\externs_tools``` by running build.sh
4. This code is built using pytorch 1.7 with CUDA 10.0 and tested on Ubuntu 18.04 with Python 3.7
### Dataset 
1. We use the [[Acronym]](https://github.com/NVlabs/acronym#using-the-full-acronym-dataset) to generate the dataset, we modify the grasp dataset to assign each grasp configuration an unique corresponding grasp point
   1. Convert the ShapeNet meshes to watertight requires ManifoldPlus. The newer version of ManifoldPlus can be [[found]](https://github.com/hjwdzh/ManifoldPlus)
   2. after the compling, we can simpliy use 
   
   ```
   mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j8
    cd build
    ./ManifoldPlus --input input.obj --output output.obj 
   ```  
2. the dataset used in this project can be founded in [[link]]( https://drive.google.com/drive/folders/1EkaV74ggHVMgiIsSMiEl1DxPoDOt5ndO?usp=sharing), and uses the tool h5ls can see the details of h5 file

### Configuration
The configuration file for defining the PCTMA-Net parameter is located in  ```transformergrap/tp_models/PTnet_grasp.yaml```
### Train 
run the train demo in the grasp network by setting the value ``` --train True --evaluate False````
```
cd script
bash run_liegrasp_net.sh 
```
### License
This project Code is released under the GPLv2 License (refer to the LICENSE file for details).
