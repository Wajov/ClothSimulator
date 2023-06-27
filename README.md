# Dependencies

- OpenGL
- GLAD (already in lib directory)
- GLFW
- jsoncpp
- Eigen
- CUDA

Note: If you're using Windows, vspkg is recommended to install dependencies.

# Build

Run the following command in this directory:

```key
mkdir build
cd build
cmake ..
make all
```

# Usage

The simulator has 5 different modes. All the command mentioned should be run in this directory. Drop the --gpu parameter if you want CPU simulation.

## Simulate

Simulate and display according to a configuration file.

```key
./build/ClothSimulator simulate [config_file] --gpu
```

For example:

```key
./build/ClothSimulator simulate conf/sphere.json --gpu
```

## Simulate (Offline)

Similar to simulate mode, but will save cloth mesh for every frame to output directory.

```key
./build/ClothSimulator simulate_offline [config__ile] [output_dir] --gpu
```

For example:

```key
./build/ClothSimulator simulate_offline conf/sphere.json output/sphere --gpu
```

## Resume

Resume and display a halted offline simulation.

```key
./build/ClothSimulator resume [output_dir] --gpu
```

## Resume (Offline)

Similar to resume mode, but will save cloth mesh for every frame to output directory.

```key
./build/ClothSimulator resume_offline [output_dir] --gpu
```

## Replay

Replay simulation result according to a output directiry. This mode has no GPU mode.

```key
./build/ClothSimulator replay [output_dir]
```