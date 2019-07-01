import os
from glob import glob
import re
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import pydicom
import PIL.Image
from pydicom_PIL import get_LUT_value
import imantics
import json


class INbreastDataset:
  def __init__(self, inbreast_dir, output_dir):
    self.inbreast_dir = inbreast_dir
    self.output_dir = output_dir
    # used dataframes
    self.dicoms_df = None
    self.cases_df = None
    self.annotations_df = None
    self.dataset = None

  def read_dicoms(self):
    in_dicom_dir = os.path.join(self.inbreast_dir, 'AllDICOMs')
    dicom_fns = glob(os.path.join(in_dicom_dir, '*.dcm'))
    total = len(dicom_fns)
    readed = 0
    dicoms = []
    for dicom_fn in dicom_fns:
      dicom_fn_basename = os.path.basename(dicom_fn)
      # infer case_id and patient_id from filename
      match = re.match('^(.+?)_(.+?)_MG_.+?_.+?_ANON\.dcm$', dicom_fn_basename)
      if not match:
        print('DICOM case filename not recognized: %s' % dicom_fn)
        continue
      case_id, patient_id = match.groups()
      case_id = int(case_id)
      # read dicom and get rows and columns
      dicom = pydicom.dcmread(dicom_fn)
      rows, columns = dicom.Rows, dicom.Columns
      # add case
      dicoms.append([patient_id, case_id, rows, columns])
      readed += 1
      print('\r%d %d/%d' % (case_id, readed, total), end="")
    print('\rreaded %d dicom files' % readed)
    print('')
    # create dataset
    dicoms_df = pd.DataFrame(dicoms, columns=['patient_id', 'case_id', 'rows', 'columns'])
    dicoms_df.set_index('case_id', inplace=True)
    self.dicoms_df = dicoms_df

  def read_cases(self, pathology_filter=None):
    cases_meta_fn = os.path.join(self.inbreast_dir, 'INbreast.csv')
    # read cases meta (csv)
    cases_meta_df = pd.read_csv(cases_meta_fn, sep=';')
    cases_meta_df.drop(columns=['Patient ID', 'Patient age', 'Acquisition date'], inplace=True)
    cases_meta_df.rename(columns={'Laterality': 'laterality', 'View': 'view', 'Acquisition date': 'acquisition_date', 'File Name': 'case_id', 'ACR': 'breast_density', 'Bi-Rads': 'birads_specific'}, inplace=True)
    cases_meta_df['birads'] = cases_meta_df['birads_specific'].apply(lambda x: int(x[0]))
    cases_meta_df['pathology'] = cases_meta_df['birads'].apply(lambda x: self.__infer_pathology(x).title())
    cases_meta_df.set_index('case_id', inplace=True)
    print('found %d cases metadata' % len(cases_meta_df))
    # create dataset
    cases_df = cases_meta_df.join(self.dicoms_df)
    print('matched %d cases/dicoms' % len(cases_df))
    # filter cases
    pathology_filter = [x.title() for x in (pathology_filter or [])]
    if len(pathology_filter) > 0:
      cases_df = cases_df[cases_df.pathology.isin(pathology_filter)]
    print('using %d cases' % len(cases_df))
    print()
    self.cases_df = cases_df

  def process_annotations(self, gen_calc=False, gen_mass=False, bbox_area_filter=None):
    in_annotations_dir = os.path.join(self.inbreast_dir, 'AllXML')
    total = len(self.cases_df)
    cases_processed = 0
    annotations_processed = 0
    annotations = []
    for case_id, case_data in self.cases_df.iterrows():
      pathology = case_data['pathology']
      case_annotation_fn = os.path.join(in_annotations_dir, '%s.xml' % case_id)
      case_annotations = self.__parse_annotations(case_annotation_fn, gen_calc=gen_calc, gen_mass=gen_mass)
      for case_annotation in case_annotations:
        annotation_id, abnormality, points_x, points_y = case_annotation
        bbox_area = (np.max(points_x) - np.min(points_x)) * (np.max(points_y) - np.min(points_y))
        if bbox_area >= bbox_area_filter:
          category = '%s%s' % (abnormality.title(), pathology.title())
          points = self.__join_points(points_x, points_y)
          annotations.append([case_id, annotation_id, category, points])
        annotations_processed += 1
      cases_processed += 1
      print('\r%d %d/%d' % (case_id, cases_processed, total), end="")
    # create dataset
    annotations_df = pd.DataFrame(annotations, columns=['case_id', 'annotation_id', 'category', 'points'])
    annotations_df.set_index('case_id', inplace=True)
    annotations_df = annotations_df.join(self.cases_df)
    print('\rprocessed %d annotations' % annotations_processed)
    print('using %d annotations' % len(annotations_df))
    print()
    self.annotations_df = annotations_df

  def save(self, all_images=True, convert_dicoms=True):
    self.prepare_dataset(all_images=all_images)
    self.save_dataset()
    if convert_dicoms:
      self.convert_dicoms()

  def prepare_dataset(self, all_images=True):
    dataset = imantics.Dataset('INbreast')
    categories_names = list(self.annotations_df.category.unique())
    categories = {name: imantics.Category(name) for name in categories_names}
    total = len(self.annotations_df)
    prepared = 0
    for case_id, annotations_data in self.annotations_df.groupby('case_id'):
      width = annotations_data['columns'].unique()[0]
      height = annotations_data['rows'].unique()[0]
      image = imantics.Image(id=case_id, width=width, height=height)
      image.file_name = '%d.png' % case_id
      annotations = []
      for _, annotation_data in annotations_data.iterrows():
        annotation_id = annotation_data['annotation_id']
        category = categories[annotation_data['category']]
        points = annotation_data['points']
        annotation = imantics.Annotation.from_polygons([points], category=category)
        annotations.append(annotation)
      image.add(annotations)
      dataset.add(image)
      prepared += 1
      print('\r%d %d/%d' % (case_id, prepared, total), end="")
    print('\rprepared %d images' % prepared)
    print()
    self.dataset = dataset

  def save_dataset(self):
    print('\rsaving dataset', end="")
    data = self.dataset.coco()
    with open('teste.json', 'w') as f:
        json.dump(data, f)
    print('\rcoco dataset saved')

  def convert_dicoms(self):
    pass
    # out_images_dir = os.path.join(output_dir, 'JPEGImages')
    # # create dirs if converting
    # os.makedirs(self.out_images_dir, exist_ok=True)
    # # convert
    # out_fn = os.path.join(self.out_images_dir, '%d.png' % case_id)
    # pixel_array = dicom.pixel_array
    # # inbreast doesnt have window date, so calculate
    # pixel_array_min, pixel_array_max = np.min(pixel_array), np.max(pixel_array)
    # window = pixel_array_max - pixel_array_min
    # level = pixel_array_min + window // 2
    # pixel_array = get_LUT_value(pixel_array, (window,), (level,))
    # image = PIL.Image.fromarray(pixel_array).convert('L')
    # # save
    # image.save(out_fn)
    # converted += 1


  def __infer_pathology(self, birads):
    if birads >= 1 and birads <= 3:
      return 'Benign'
    elif birads > 3:
      return 'Malignant'

  def __parse_annotations(self, annotation_fn, gen_calc=False, gen_mass=False):
    if not os.path.exists(annotation_fn):
      return []
    annotations = []
    tree = ET.parse(annotation_fn)
    root = tree.getroot()
    for elem_roi in root.findall('./dict/array/dict/array/dict'):
      abnormality = elem_roi[15].text
      if not ((abnormality == 'Calcification' and gen_calc) or (abnormality == 'Mass' and gen_mass)):
        continue
      annotation_id = int(elem_roi[7].text)
      points_x = []
      points_y = []
      for elem_point in elem_roi[21].findall('./'):
        x, y = elem_point.text.strip('()').split(',')
        points_x.append(round(float(x), 1))
        points_y.append(round(float(y), 1))
      annotations.append([annotation_id, abnormality, points_x, points_y])
    return annotations

  def __join_points(self, xs, ys):
    result = []
    for x, y in zip(xs, ys):
      result.append(x)
      result.append(y)
    return result
