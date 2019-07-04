import mammo_dataset
import json


inbreast_dir = '/Users/lpires/Developer/dl/INbreast'
output_dir = './tmp'

mds = mammo_dataset.INbreastDataset(inbreast_dir, output_dir)
mds.read_dicoms()
mds.read_cases(pathology_filter=[])

print(mds.cases_df.groupby(['view', 'width', 'height']).size())
print()

mds.process_annotations(gen_calc=True,
                        gen_mass=True,
                        gen_asymmetry=True,
                        gen_distortion=True,
                        gen_spiculated=True,
                        bbox_area_filter=(0))

print('annotations:', len(mds.annotations_df))
print('images:', len(mds.annotations_df.dicom_fn.unique()))
print(mds.annotations_df.groupby('category').size())
print()

all_images = False
convert_dicoms = False

# mds.save(all_images=all_images, convert_dicoms=convert_dicoms)

# mds.prepare_dataset(all_images=all_images)
# mds.draw_annotations('./tmp_annotations/')
