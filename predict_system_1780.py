import tools.infer.utility as utility# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import os
import cv2
import copy
import numpy as np
import time
import logging
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger
from tools.infer.utility import get_rotate_crop_image, draw_ocr_box_txt_user

logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0
        print("2")

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno + self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        print("3")
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        # print(dt_boxes)

        logger.debug("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes_other = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes_other)):
            tmp_box = copy.deepcopy(dt_boxes_other[bno])

            tmp_box = np.array(tmp_box, dtype=np.float32)
            tmp_box = cv2.minAreaRect(tmp_box)

            points = sorted(list(cv2.boxPoints(tmp_box)), key=lambda x: x[0])
            index_1, index_2, index_3, index_4 = 0, 1, 2, 3
            if points[1][1] > points[0][1]:
                index_1 = 0
                index_4 = 1
            else:
                index_1 = 1
                index_4 = 0
            if points[3][1] > points[2][1]:
                index_2 = 2
                index_3 = 3
            else:
                index_2 = 3
                index_3 = 2
            box = [
                points[index_1], points[index_2], points[index_3], points[index_4]
            ]

            center = np.array(
                [points[index_1].tolist(), points[index_2].tolist(), points[index_3].tolist(), points[index_4].tolist()]
                , dtype="float32")

            print(box)
            print(type(box))

            img_crop = get_rotate_crop_image(ori_im, box, center)
            img_crop_list.append(img_crop)
        # if self.use_angle_cls and cls:
        #     img_crop_list, angle_list, elapse = self.text_classifier(
        #         img_crop_list)
        #     logger.debug("cls num  : {}, elapse : {}".format(
        #         len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # if self.args.save_crop_res:
        #     self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
        #                            rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def main(args, img):
    # image_file_list = get_image_file_list(args.image_dir)
    # image_file_list = image_file_list[args.process_id::args.total_process_num]
    global draw_img
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = 0.85

    total_time = 0
    cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    _st = time.time()

    starttime = time.time()
    print("1")
    dt_boxes, rec_res = text_sys(img)
    rec_res_new = []
    elapse = time.time() - starttime
    total_time += elapse

    for text, score in rec_res:
        if score >= drop_score:
            logger.debug("{}, {:.3f}".format(text, score))
            rec_res_new.append([text, score])

    if is_visualize:
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes = dt_boxes
        txts = [rec_res[i][0] for i in range(len(rec_res))]
        scores = [rec_res[i][1] for i in range(len(rec_res))]

        draw_img = draw_ocr_box_txt_user(
            image,
            boxes,
            txts,
            scores,
            drop_score=drop_score,
            font_path=font_path)
        # draw_img = draw_text_det_res(boxes, image)
        # draw_img_save_dir = args.draw_img_save_dir
        # os.makedirs(draw_img_save_dir, exist_ok=True)

    return rec_res_new, draw_img, elapse

# if __name__ == "__main__":
#     args = utility.parse_args()
#     if args.use_mp:
#         p_list = []
#         total_process_num = args.total_process_num
#         for process_id in range(total_process_num):
#             cmd = [sys.executable, "-u"] + sys.argv + [
#                 "--process_id={}".format(process_id),
#                 "--use_mp={}".format(False)
#             ]
#             p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
#             p_list.append(p)
#         for p in p_list:
#             p.wait()
#     else:
#         main(args)
