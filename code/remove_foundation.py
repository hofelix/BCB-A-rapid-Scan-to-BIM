import numpy
import bpy
import os
import difflib


obj_path = os.path.abspath('D:\Program\RANSAC_EDGE\obj')
save_path = os.path.abspath('D:\Program\RANSAC_EDGE\obj_finish')


def main():
    for i in range(100):
        join_obj(i)

def join_obj(i):
    obj_list = []
    import_obj = str(i)+'.obj'
    import_path = os.path.join(obj_path,import_obj)
    export_path = os.path.join(save_path,import_obj)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.obj(filepath = import_path)
    bpy.data.objects.remove(bpy.data.objects['Ground_Motion'])
    for obj in bpy.data.objects:
        obj_list.append(obj.name)
    foundation_list = difflib.get_close_matches('Foundation',obj_list,100,cutoff=0.6)
    obj_list = []
    for f in foundation_list:
        bpy.data.objects.remove(bpy.data.objects[f])
    bpy.context.view_layer.objects.active = bpy.data.objects[0]
    bpy.ops.object.join()
    bpy.ops.export_scene.obj(filepath = export_path)


if __name__ == '__main__':
    main()
