import mammo_dataset
import json


inbreast_dir = '/Users/lpires/Developer/dl/INbreast'
output_dir = './tmp'

mds = mammo_dataset.INbreastDataset(inbreast_dir, output_dir)
mds.read_dicoms()
mds.read_cases(pathology_filter=[])
mds.process_annotations(gen_calc=False, gen_mass=True, bbox_area_filter=(32 * 32))

print(mds.annotations_df.groupby('category').size())
print()

all_images = False
convert_dicoms = False

mds.save(all_images=all_images, convert_dicoms=convert_dicoms)

# mds.prepare_dataset(all_images=all_images)
# mds.export_annotations()
