import argparse
import os
from glob import glob
from math import floor, ceil
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import pydicom


parser = argparse.ArgumentParser(description='Convert INbreast dataset annotations to PASCAL VOC 2007 structure')
parser.add_argument('inbreast_dir', help='input INbreast dataset base dir')
parser.add_argument('output_dir')
parser.add_argument('--gen_mass', help='generate annotations for masses', action='store_true')
parser.add_argument('--gen_calc', help='generate annotations for microcalcifications', action='store_true')
parser.add_argument('--calc_padding', help='padding added to microcalcifications clusters', type=int, default=20)
parser.add_argument('--pathology_filter', action='append', help='filter by pathology')
parser.add_argument('--pathology_in_name', help='add pathology do annotation name', action='store_true')

args = parser.parse_args()

xml_dir = os.path.join(args.inbreast_dir, 'AllXML')
dcm_dir = os.path.join(args.inbreast_dir, 'AllDICOMs')
gen_mass = args.gen_mass
gen_calc = args.gen_calc

###
# prepare data
###

# cases

print('reading dataset cases...')

cases_fn = os.path.join(args.inbreast_dir, 'INbreast.csv')
cases_df = pd.read_csv(cases_fn, sep=';')
cases_df.drop(columns=['Patient ID', 'Patient age', 'Acquisition date'], inplace=True)
cases_df.rename(columns={'Laterality': 'laterality', 'View': 'view', 'Acquisition date': 'acquisition_date', 'File Name': 'id', 'ACR': 'breasy_density', 'Bi-Rads': 'birads_specific'}, inplace=True)
cases_df['birads'] = cases_df['birads_specific'].apply(lambda x: x[0])

print('found %d cases' % len(cases_df))


def pathology(birads):
  if birads >= 1 and birads <= 3:
    return 'BENIGN'
  elif birads > 3:
    return 'MALIGNANT'
cases_df['pathology'] = cases_df['birads'].apply(lambda x: pathology(int(x)))

pathology_filter = [x.upper() for x in (args.pathology_filter or [])]
if len(pathology_filter) > 0:
  cases_df = cases_df[cases_df.pathology.isin(pathology_filter)]

print('using %d cases' % len(cases_df))
print('')


# rois

print('reading rois...')

rois = []
for id in cases_df.id:
  roi_fn = os.path.join(xml_dir, '%d.xml' % id)
  if not os.path.exists(roi_fn):
    print('File not found %s' % roi_fn)
    continue
  tree = ET.parse(roi_fn)
  root = tree.getroot()
  for elem_roi in root.findall('./dict/array/dict/array/dict'):
    abnormality = elem_roi[15].text
    if not ((abnormality == 'Mass' and gen_mass) or (abnormality == 'Calcification' and gen_calc)):
      continue
    index = elem_roi[7].text
    points_x = []
    points_y = []
    for elem_point in elem_roi[21].findall('./'):
      x, y = elem_point.text.strip('()').split(',')
      points_x.append(int(float(x)))
      points_y.append(int(float(y)))
    rois.append([id, int(index), abnormality, min(points_x), min(points_y), max(points_x), max(points_y)])
rois_df = pd.DataFrame(rois, columns=['id', 'index', 'abnormality', 'min_x', 'min_y', 'max_x', 'max_y'])

print('using %d rois' % len(rois_df))
print('')


# metadata
# TODO: gerar número do paciente

print('reading dicoms metadata...')

dcm_data = []
for id in rois_df.id.unique():
  print('\r%s' % id, end="")
  dcm_fns = glob(os.path.join(dcm_dir, str(id) + '*.dcm'))
  assert len(dcm_fns) == 1
  dcm_fn = dcm_fns[0]
  dcm = pydicom.dcmread(dcm_fn)
  dcm_data.append([id, dcm.Rows, dcm.Columns, os.path.basename(dcm_fn)])
dcm_df = pd.DataFrame(dcm_data, columns=['id', 'rows', 'columns', 'dcm_fn'])

print('')
print('found %s dicom files' % len(dcm_df))
print('')

# join dataframes
# TODO: usar left para no caso de não haver anotações
df = rois_df.set_index('id').join(cases_df.set_index('id')).join(dcm_df.set_index('id'))

# expand calcifications bounding boxes
def expand_calcifications(r, padding):
  if r['abnormality'] != 'Calcification':
    return r
  r['min_x'] = max(0, r['min_x'] - padding)
  r['min_y'] = max(0, r['min_y'] - padding)
  r['max_x'] = min(r['max_x'] + padding, r['columns'])
  r['max_y'] = min(r['max_y'] + padding, r['rows'])
  return r
df = df.apply(lambda r: expand_calcifications(r, args.calc_padding), axis=1)


###
# write annotations
###

# sobre o pascal voc: o que fazer com pose e truncated?
# sobre o inbreast: o que fazer com acr, birads?

pathology_in_name = args.pathology_in_name
def to_xml(index, grouped):
  folder = 'VOC2007'
  filename = str(index) + '.jpg'
  width = grouped['columns'].min()
  height = grouped['rows'].min()
  laterality = grouped['laterality'].min()
  view = grouped['view'].min()
  breast_density = grouped['breasy_density'].min()
  birads = grouped['birads'].min()
  birads_specific = grouped['birads_specific'].min()
  abnormality = grouped['abnormality'].min()
  pathology = grouped['pathology'].min()
  name = '%s%s' % (abnormality, pathology) if pathology_in_name else abnormality
  # iterate over cases
  objs = []
  for _, row in grouped.iterrows():
    obj = '\n'.join([
      '    <object>',
      '        <name>%s</name>' % name,
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
    '    <inbreast_breast_density>%s</inbreast_breast_density>' % breast_density,
    '    <inbreast_birads>%s</inbreast_birads>' % birads,
    '    <inbreast_birads_specific>%s</inbreast_birads_specific>' % birads_specific,
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
annotations_dir = os.path.join(args.output_dir, 'Annotations')
os.makedirs(annotations_dir, exist_ok=True)
print('writing annotations...')
wrote = 0
for index, grouped in df.groupby('id'):
    print('\r%s' % index, end="")
    xml = to_xml(index, grouped)
    xml_fn = os.path.join(annotations_dir, str(index) + '.xml')
    with open(xml_fn, 'w') as f:
        f.write(xml)
    wrote += 1
print('')
print('')
print('wrote %d annotations files' % wrote)

# write image sets

trainval = [str(x) for x in list(np.unique(df.index.values))]
test = trainval

image_sets_dir = os.path.join(args.output_dir, 'ImageSets', 'Main')
trainval_fn = os.path.join(image_sets_dir, 'trainval.txt')
test_fn = os.path.join(image_sets_dir, 'test.txt')
os.makedirs(image_sets_dir, exist_ok=True)

with open(trainval_fn, 'w') as f:
  f.write('\n'.join(trainval))
with open(test_fn, 'w') as f:
  f.write('\n'.join(test))

# images

jpeg_images_dir = os.path.join(args.output_dir, 'JPEGImages')
os.makedirs(jpeg_images_dir, exist_ok=True)
