import os.path as osp
from glob import glob
import pydicom
import xml.etree.ElementTree as ET
import cv2
import re
import numpy as np
from operator import itemgetter


def parse_annotations(annotation_fn):
  annotations = []
  tree = ET.parse(annotation_fn)
  root = tree.getroot()
  elem_check = root.findall('./dict/key')
  if len(elem_check) >= 2 and elem_check[1].text == 'DataSummary':
    elem_array = root.findall('./dict/array')[1]
  else:
    elem_array = root.findall('./dict/array/dict/array/dict/array')[1]
  result = []
  for i, elem_roi in enumerate(elem_array.findall('./string')):
    x, y = elem_roi.text[1:-1].split(', ')
    x = int(float(x))
    y = int(float(y))
    result.append((x, y,))
  return result

def assert_ascending(arr):
  if 0 <= len(arr) <= 1:
    return True
  previous = arr[0]
  for i in range(1, len(arr)):
    current = arr[i]
    if previous > current:
      return False
    previous = current
  return True

def prepare_polygon(pts, width, height):
  if len(pts) == 0:
    return []
  first_i = 0
  last_i = len(pts) - 1
  pts[first_i] = (pts[first_i][0], 0)
  if laterality == 'L':
    pts[last_i] = (0, pts[last_i][1])
    pts.append((0, 0))
  elif laterality == 'R':
    pts[last_i] = (width - 1, pts[last_i][1])
    pts.append((width - 1, 0))
  return pts

def show_image(img):
  cv2.imshow('teste', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


inbreast_dir = '/Users/lpires/Developer/dl/INbreast'
annotations_dir = osp.join(inbreast_dir, 'PectoralMuscle/Pectoral Muscle XML')
dicoms_dir = osp.join(inbreast_dir, 'AllDICOMs')
images_dir = '/Users/lpires/Developer/dl/dummy-scratch/out/images'
scale = 0.2
debug = True

# cases_ids = [20588046, 20588072]
cases_ids = sorted([int(osp.basename(x).split('_')[0]) for x in glob(osp.join(annotations_dir, '*.xml'))])

total = len(cases_ids)
processed = 0
for case_id in cases_ids:
  warnings = []
  # check annotation
  anns_fns = glob(osp.join(annotations_dir, '*%d*.xml' % case_id))
  assert len(anns_fns) == 1
  ann_fn = anns_fns[0]
  # check dicom
  dcms_fns = glob(osp.join(dicoms_dir, '*%d*.dcm' % case_id))
  assert len(dcms_fns) == 1
  dcm_fn = dcms_fns[0]
  # check image
  imgs_fns = glob(osp.join(images_dir, '*%d*.png' % case_id))
  assert len(imgs_fns) == 1
  img_fn = imgs_fns[0]
  # check laterality
  match = re.match('^.+?_.+?_MG_(.+?)_.+?_ANON\.dcm$', dcm_fn)
  assert match is not None
  laterality = match[1]
  assert laterality in ['L', 'R']
  # parse annotations
  pts = parse_annotations(ann_fn)
  assert len(pts) > 0
  # sort y
  pts = sorted(pts, key=itemgetter(1))
  assert assert_ascending([pt[1] for pt in pts])
  # read image
  img = cv2.imread(img_fn)
  height, width, _ = img.shape
  scaled_dim = (int(height * scale), int(width * scale),)
  # prepare polygon
  pts = prepare_polygon(pts, width, height)
  pts = np.array(pts, np.int32)
  # show processed
  processed += 1
  if len(warnings) == 0:
    print('%d %d/%d' % (case_id, processed, total))
  else:
    print('%d (%s) %d/%d' % (case_id, ', '.join(warnings), processed, total))
  # create pectoral muscle mask
  pm_mask_gray = np.zeros((height, width, 1), np.uint8)
  pm_mask_gray = cv2.fillPoly(pm_mask_gray, [pts], 255)
  # create breast mask
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _ , img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  breast_mask_gray = cv2.bitwise_and(img_gray, img_gray, mask=cv2.bitwise_not(pm_mask_gray))
  # resize
  pm_mask_gray = cv2.resize(pm_mask_gray, scaled_dim)
  breast_mask_gray = cv2.resize(breast_mask_gray, scaled_dim)
  # show debug
  if debug:
    # resize image
    img_debug = np.copy(img)
    for pt in pts:
      pt_tuple = (pt[0], pt[1])
      img_debug = cv2.circle(img_debug, pt_tuple, 10, (0, 255, 255), -1)
    img_debug = cv2.resize(img_debug, scaled_dim)
    # display
    pm_mask_bgr = cv2.cvtColor(pm_mask_gray, cv2.COLOR_GRAY2BGR)
    pm_mask_bgr = cv2.bitwise_and(pm_mask_bgr, (255, 0, 0), mask=pm_mask_gray)
    breast_mask_bgr = cv2.cvtColor(breast_mask_gray, cv2.COLOR_GRAY2BGR)
    breast_mask_bgr = cv2.bitwise_and(breast_mask_bgr, (0, 0, 255), mask=breast_mask_gray)
    blended = cv2.addWeighted(img_debug, 1.0, breast_mask_bgr, 0.1, 0)
    blended = cv2.addWeighted(blended, 1.0, pm_mask_bgr, 0.1, 0)
    show_image(blended)
