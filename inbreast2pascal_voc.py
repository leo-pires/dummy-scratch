import argparse
import os
from glob import glob
from math import floor, ceil
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd
import pydicom


parser = argparse.ArgumentParser(description='Convert INbreast dataset annotations to PASCAL VOC 2007 structure')
parser.add_argument('inbreast_dir', help='input INbreast dataset base dir')
parser.add_argument('output_dir')
parser.add_argument('--gen_mass', help='generate mass annotations', action='store_true')
parser.add_argument('--gen_mcs', help='generate microcalcifications annotations', action='store_true')
parser.add_argument('--mcs_padding', help='padding added to microcalcifications clusters', type=int, default=20)

args = parser.parse_args()

xml_dir = os.path.join(args.inbreast_dir, 'AllXML')
dcm_dir = os.path.join(args.inbreast_dir, 'AllDICOMs')
dataset_fn = os.path.join(args.inbreast_dir, 'INbreast.csv')
gen_mass = args.gen_mass
gen_mcs = args.gen_mcs

###
# prepare data
###

print('reading dataset cases...')

dataset_df = pd.read_csv(dataset_fn, sep=';')
dataset_df.drop(columns=['Patient ID', 'Patient age', 'Acquisition date'], inplace=True)
dataset_df.rename(columns={'File Name': 'id', 'Laterality': 'laterality', 'View': 'view', 'ACR': 'acr', 'Bi-Rads': 'birads'}, inplace=True)

print('reading dataset cases... done')


print('reading rois...')

rois = []
for roi_fn in glob(os.path.join(xml_dir, '*.xml')):
	id = int(os.path.splitext(os.path.basename(roi_fn))[0])
	tree = ET.parse(roi_fn)
	root = tree.getroot()
	for elem_roi in root.findall('./dict/array/dict/array/dict'):
		abnormality_type = elem_roi[15].text
		if not ((abnormality_type == 'Mass' and gen_mass) or (abnormality_type == 'Calcification' and gen_mcs)):
			continue
		index = elem_roi[7].text
		points_x = []
		points_y = []
		for elem_point in elem_roi[21].findall('./'):
			x, y = elem_point.text.strip('()').split(',')
			points_x.append(int(float(x)))
			points_y.append(int(float(y)))
		rois.append([id, int(index), abnormality_type, min(points_x), min(points_y), max(points_x), max(points_y)])
rois_df = pd.DataFrame(rois, columns=['id', 'index', 'type', 'min_x', 'min_y', 'max_x', 'max_y'])

print('reading rois... done')


print('reading dicoms metadata...')

dcm_data = []
for id in rois_df.id.unique():
	dcm_fns = glob(os.path.join(dcm_dir, str(id) + '*.dcm'))
	assert len(dcm_fns) == 1
	dcm_fn = dcm_fns[0]
	dcm = pydicom.dcmread(dcm_fn)
	dcm_data.append([id, dcm.Rows, dcm.Columns, os.path.basename(dcm_fn)])
dcm_df = pd.DataFrame(dcm_data, columns=['id', 'rows', 'columns', 'dcm_fn'])

print('reading dicoms metadata... done')


# join dataframes
df = rois_df.set_index('id').join(dataset_df.set_index('id')).join(dcm_df.set_index('id'))

# expand calcifications bounding boxes
def expand_calcifications(r, padding):
	if r['type'] != 'Calcification':
		return r
	r['min_x'] = max(0, r['min_x'] - padding)
	r['min_y'] = max(0, r['min_x'] - padding)
	r['max_x'] = min(r['max_x'] + padding, r['columns'])
	r['max_y'] = min(r['max_y'] + padding, r['rows'])
	return r
df = df.apply(lambda r: expand_calcifications(r, args.mcs_padding), axis=1)


###
# write annotations
###

# sobre o pascal voc: o que fazer com pose e truncated?
# sobre o inbreast: o que fazer com acr, birads?

def to_xml(index, grouped):
	folder = 'VOC2007'
	filename = str(index) + '.jpg'
	width = grouped['columns'].min()
	height = grouped['rows'].min()
	laterality = grouped['laterality'].min()
	view = grouped['view'].min()
	acr = grouped['acr'].min()
	birads = grouped['birads'].min()

	objs = []
	for _, row in grouped.iterrows():
		obj = '\n'.join([
			'    <object>',
			'        <name>%s</name>' % row['type'],
			'        <difficult>0</difficult>',
			'        <bndbox>',
			'            <xmin>%d</xmin>' % row['min_x'],
			'            <ymin>%d</ymin>' % row['min_y'],
			'            <xmax>%d</xmax>' % row['max_x'],
			'            <ymax>%d</ymax>' % row['max_y'],
			'        </bndbox>',
			'    </object>'
		])
		objs.append(obj)
	doc = '\n'.join([
		'<annotation>',
		'    <folder>%s</folder>' % folder,
		'    <filename>%s</filename>' % filename,
		'    <inbreast_laterality>%s</inbreast_laterality>' % laterality,
		'    <inbreast_view>%s</inbreast_view>' % view,
		'    <inbreast_acr>%s</inbreast_acr>' % acr,
		'    <inbreast_birads>%s</inbreast_birads>' % birads,
		'\n'.join([
			'    <size>',
			'        <width>%s</width>' % width, 
			'        <height>%s</height>' % height, 
			'        <width>3</width>',
			'    </size>'
		]),
		'    <segmented>0</segmented>',
		'\n'.join(objs),
		'</annotation>'
	])

	return doc

# write annotations xml
print('writing annotations...')
print('', end="")
for index, grouped in df.groupby('id'):
    print('\r%s' % index, end="")
    xml = to_xml(index, grouped)
    xml_fn = os.path.join(args.output_dir, 'Annotations', str(index) + '.xml')
    with open(xml_fn, 'w') as f:
        f.write(xml)
print('done!')
