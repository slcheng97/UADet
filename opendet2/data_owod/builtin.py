import os

from .voc_coco import register_pascal_voc_owod
from detectron2.data import MetadataCatalog


def register_all_voc_coco_owod(root):
    SPLITS = [
        # VOC_COCO_openset
        ("M_OWODB_t1_train", "M-OWODB", "M-OWODB", "t1"),
        ("M_OWODB_t2_train", "M-OWODB", "M-OWODB", "t2"),
        ("M_OWODB_t2_ft", "M-OWODB", "M-OWODB", "t2_ft"),
        ("M_OWODB_t3_train", "M-OWODB", "M-OWODB", "t3"),
        ("M_OWODB_t3_ft", "M-OWODB", "M-OWODB", "t3_ft"),
        ("M_OWODB_t4_train", "M-OWODB", "M-OWODB", "t4"),
        ("M_OWODB_t4_ft", "M-OWODB", "M-OWODB", "t4_ft"),
        ("M_OWODB_test", "M-OWODB", "M-OWODB", "test"),
        ("S_OWODB_t1_train", "S-OWODB", "S-OWODB", "t1"),
        ("S_OWODB_t2_train", "S-OWODB", "S-OWODB", "t2"),
        ("S_OWODB_t2_ft", "S-OWODB", "S-OWODB", "t2_ft"),
        ("S_OWODB_t3_train", "S-OWODB", "S-OWODB", "t3"),
        ("S_OWODB_t3_ft", "S-OWODB", "S-OWODB", "t3_ft"),
        ("S_OWODB_t4_train", "S-OWODB", "S-OWODB", "t3"),
        ("S_OWODB_t4_ft", "S-OWODB", "S-OWODB", "t4_ft"),
        ("S_OWODB_test", "S-OWODB", "S-OWODB", "test"),
    ]
    for name, dirname, super_split, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc_owod(name, os.path.join(root, dirname), super_split, split, year)
        # register_pascal_voc_owod(name, root, super_split, split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


if __name__.endswith(".builtin"):
    # Register them all under "./datasets"
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_voc_coco_owod(_root)