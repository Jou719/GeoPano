import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from geopano.panorama import PanoramaToPerspective
from geopano.panorama_geometry import split_panorama_image
from files_inference import save_cloud, collect_images
from pathlib import Path
p = PanoramaToPerspective(output_resolution=518, view_type='10views')
views = split_panorama_image(np.zeros((1440,2880,3),np.uint8),p.mapping,True)
assert len(views)==10 and views[0].shape==(518,518,3)
x,y=np.meshgrid(np.arange(518),np.arange(518))
k=p.np_intrisics
rays=np.stack([(x-k[0,2])/k[0,0],(y-k[1,2])/k[1,1],np.ones_like(x)],-1)
rays=rays/np.linalg.norm(rays,axis=-1,keepdims=True)
world=[(rays @ np.linalg.inv(ext)[:3,:3].T).astype(np.float32) for ext in p.np_extrinsics]
xyz,_,mask=p.fuse_perspective_pointmaps_to_pano(world,center_weighted=True)
assert mask.mean()>.99,mask.mean()
assert np.max(np.abs(np.linalg.norm(xyz[mask],axis=-1)-1))<1e-4
(ROOT / 'outputs').mkdir(exist_ok=True)
save_cloud(ROOT / 'outputs/geometry_check.ply', xyz[mask][::1000], np.ones_like(xyz[mask][::1000])*.5)
example_images = collect_images(ROOT / 'examples')
assert example_images and all(image.is_file() for image in example_images)
print('PASS: ten-view split, unit-sphere fusion, >99% coverage, finite XYZ, PLY export, example discovery')
