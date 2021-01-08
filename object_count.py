from pycocotools.coco import COCO
from pycocotools.mask import area


coco = COCO('/shared/xudongliu/bdd100k/labels/ins_seg/ins_seg_val.json')
total_object_count = 0
small_count = 0
medium_count = 0
large_count = 0
img_ids = coco.getImgIds()
for img_id in img_ids:
    anno_ids = coco.getAnnIds(imgIds = [img_id])
    annos = coco.loadAnns(anno_ids)
    total_object_count += len(annos)
    for anno in annos:
        rle = coco.annToRLE(anno)
        anno_area = area(rle)
        if anno_area < 32 * 32:
            small_count += 1
        elif anno_ids < 96 *96:
            medium_count += 1
        else:
            large_count += 1
print(total_object_count, '\n', small_count, '\n', medium_count, '\n', large_count)