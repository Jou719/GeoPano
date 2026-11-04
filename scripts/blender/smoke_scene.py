"""Headless Blender smoke test for a deterministic editable room."""
import bpy
import math
from pathlib import Path

out = Path(__file__).resolve().parents[2] / 'outputs' / 'blender_smoke'
out.mkdir(parents=True, exist_ok=True)
# Reset.
bpy.ops.wm.read_factory_settings(use_empty=True)

def box(name, location, scale, color):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    mat = bpy.data.materials.new(name + '_material')
    mat.diffuse_color = (*color, 1)
    obj.data.materials.append(mat)
    return obj
box('floor', (0, 0, -0.05), (3, 3, 0.05), (0.35, 0.35, 0.35))
box('back_wall', (0, 3, 1.5), (3, 0.05, 1.5), (0.7, 0.7, 0.7))
box('left_wall', (-3, 0, 1.5), (0.05, 3, 1.5), (0.6, 0.65, 0.7))
box('table', (0, 0.5, 0.8), (1.0, 0.6, 0.05), (0.5, 0.25, 0.1))
# Equirectangular camera looking along +Y, z-up.
bpy.ops.object.camera_add(location=(0, 0, 1.6))
cam = bpy.context.object
cam.name = 'panorama_camera'
cam.data.type = 'PANO'
cam.data.lens = 18
cam.data.panorama_type = 'EQUIRECTANGULAR'
cam.rotation_euler = (math.pi / 2, 0, 0)
bpy.context.scene.camera = cam
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 4
scene.world = bpy.data.worlds.new('World')
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.5, 0.5, 0.5, 1)
scene.render.resolution_x = 256
scene.render.resolution_y = 128
scene.render.resolution_percentage = 100
scene.render.filepath = str(out / 'room_equirectangular.png')
scene.render.image_settings.file_format = 'PNG'
bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'room.blend'))
scene.render.filepath = str(out / 'room.glb')
bpy.ops.export_scene.gltf(filepath=str(out / 'room.glb'), export_format='GLB')
print('PASS: rendered', scene.render.filepath)
