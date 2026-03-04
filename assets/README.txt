Put model files in this folder.

Built-in sample models:
- model_cube.obj
- model_cone.obj
- model_sphere.obj

Example run:
- python main.py --model assets/model_cube.obj
- python main.py --model assets/model_cone.obj
- python main.py --model assets/model_sphere.obj

If model file is missing, the app uses a fallback cube.

Supported directly by trimesh: .glb/.gltf/.obj (and other trimesh formats).
Extended support (.dae/.3ds and similar) works via Assimp CLI fallback.

Example particle config:
- particle_config_example.json
