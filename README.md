# Ro.Bo.
Ro.Bo.: Automatic Rigging of Human Body
1) Colab to execute PIFU and RigNet to obtain the glb file with the script blender_save_gltf.py with the command:
´´´
blender -P blender_save_gltf.py -- -i /path_to/result_ori.obj -r /path_to/result_ori_rig.txt -o /path_to/result.glb
´´´
2) Animation with Panda3D on the folder animatepy with run.sh
