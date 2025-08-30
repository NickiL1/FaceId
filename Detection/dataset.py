import numpy as np
from PIL import Image
import os
from AnchorBoxes import *




def get_labels(dir_path, images_path):
    folder_path = os.path.join(dir_path, "train")
    label_file = os.path.join(folder_path, "label.txt")
    with open(label_file, "r") as f:
        lines = f.readlines()
    labels = []
    for ind,line in enumerate(lines):
        parts = line.split(" ")
        if len(parts) == 2:
            if ind != 0:
                labels.append((curr_path, tf.convert_to_tensor(curr_data, dtype=tf.float32)))
            curr_path = parts[1].strip()
            curr_data = []
            with Image.open(f"{images_path}/{curr_path}") as img:
                H, W = img.height, img.width
        else:
            parts_num = np.array(parts, dtype=np.float32)
            bbox = parts_num[:4]
            x1, y1, w, h = bbox / [W, H, W, H]
            cx, cy = x1 + w / 2, y1 + h / 2
            lnd1_x, lnd1_y = parts_num[4:6] / [W, H]
            lnd2_x, lnd2_y = parts_num[7:9] / [W, H]
            lnd3_x, lnd3_y = parts_num[10:12] / [W, H]
            lnd4_x, lnd4_y = parts_num[13:15] / [W, H]
            lnd5_x, lnd5_y = parts_num[16:18] / [W, H]

            curr_data.append([cx, cy, w, h, lnd1_x, lnd1_y, lnd2_x, lnd2_y, lnd3_x, lnd3_y, lnd4_x, lnd4_y, lnd5_x, lnd5_y])

            if ind == len(lines)-1:
                labels.append((curr_path, tf.convert_to_tensor(curr_data, dtype=tf.float32)))

    return labels



def match_anchors_to_gt(anchors, gt_boxes, gt_landmarks, pos_thresh=0.5, neg_thresh=0.4, min_gt_iou=0.3):
    iou_mat = Iou_mat(anchors, gt_boxes)


    max_iou_per_gt = tf.reduce_max(iou_mat, axis=0)
    valid_gt_mask = max_iou_per_gt >= min_gt_iou
    gt_boxes = tf.boolean_mask(gt_boxes, valid_gt_mask)
    gt_landmarks = tf.boolean_mask(gt_landmarks, valid_gt_mask)

    if len(gt_boxes) == 0:
        return None, None, None, None

    iou_mat = Iou_mat(anchors, gt_boxes)

    max_iou = tf.reduce_max(iou_mat, axis=1)
    matched_gt_idx = tf.argmax(iou_mat, axis=1, output_type=tf.int32)

    labels = tf.fill([tf.shape(anchors)[0]], -1)
    labels = tf.where(max_iou >= pos_thresh, tf.ones_like(labels), labels)
    labels = tf.where(max_iou < neg_thresh, tf.zeros_like(labels), labels)

    iou_t = tf.transpose(iou_mat)
    num_gt = tf.shape(gt_boxes)[0]

    best_anchor_for_gt = tf.argmax(iou_t, axis=1, output_type=tf.int32)
    gt_ind = tf.range(num_gt, dtype=tf.int32)

    labels = tf.tensor_scatter_nd_update(
        labels,
        tf.expand_dims(best_anchor_for_gt, 1),
        tf.ones([num_gt], dtype=tf.int32)
    )
    matched_gt_idx = tf.tensor_scatter_nd_update(
        matched_gt_idx,
        tf.expand_dims(best_anchor_for_gt, 1),
        gt_ind
    )

    return matched_gt_idx, labels, gt_boxes, gt_landmarks


def compute_reg_targets(anchors, gt_boxes, cls_labels, matched_gt_idx, variances=[0.1,0.2]):
    matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)

    xa, ya, wa, ha = tf.unstack(anchors, axis=1)
    xt, yt, wt, ht = tf.unstack(matched_gt_boxes, axis=1)


    dx = (xt - xa) / (wa * variances[0])
    dy = (yt - ya) / (ha * variances[0])
    dw = tf.math.log(wt / wa) / variances[1]
    dh = tf.math.log(ht / ha) / variances[1]

    reg_targets = tf.stack([dx, dy, dw, dh], axis=1)
    pos_mask = tf.equal(cls_labels, 1)
    reg_targets = tf.where(tf.expand_dims(pos_mask,axis=1), reg_targets, tf.zeros_like(reg_targets))
    return reg_targets

def compute_lnd_targets(anchors, gt_landmarks, cls_labels, matched_gt_idx, variances=[0.1,0.2]):

    matched_landmarks = tf.gather(gt_landmarks, matched_gt_idx)
    cx, cy, w, h = tf.unstack(anchors, axis=1)
    cx = tf.expand_dims(cx, axis=1)
    cy = tf.expand_dims(cy, axis=1)
    w = tf.expand_dims(w, axis=1)
    h = tf.expand_dims(h, axis=1)

    x_coords = matched_landmarks[:,0::2]
    y_coords = matched_landmarks[:,1::2]

    dx = (x_coords - cx) / (w * variances[0])
    dy = (y_coords - cy) / (h * variances[0])

    landmark_targets = tf.stack([dx, dy], axis=2)
    landmark_targets = tf.reshape(landmark_targets, [tf.shape(anchors)[0],10])

    valid_x = x_coords >= 0
    valid_y = y_coords >= 0
    valid_coords5 = tf.logical_and(valid_x, valid_y)
    valid_coords10 = tf.reshape(tf.stack([valid_coords5, valid_coords5], axis=2), [tf.shape(anchors)[0],10])


    landmark_targets = tf.where(valid_coords10, landmark_targets, tf.zeros_like(landmark_targets))

    pos_mask = tf.equal(cls_labels, 1)

    landmark_targets = tf.where(tf.expand_dims(pos_mask,axis=1), landmark_targets, tf.zeros_like(landmark_targets))

    valid_coords10 = tf.cast(valid_coords10, tf.float32)
    pos_mask = tf.cast(pos_mask, tf.float32)
    pos_mask = tf.expand_dims(pos_mask, axis=1)
    landmark_mask = valid_coords10 * pos_mask
    landmark_mask = tf.reduce_all(landmark_mask > 0, axis=1)
    landmark_mask = tf.cast(landmark_mask, tf.int32)

    return landmark_targets, landmark_mask


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tfrecord(datadir, output_path, labels, anchors, shuffle=True):
    with tf.io.TFRecordWriter(output_path) as writer:
        for path, label in labels:
            img_path = os.path.join(datadir, path)
            gt_lnd = label[:, 4:]
            gt = label[:, :4]
            ind, cls, gt, gt_lnd = match_anchors_to_gt(anchors, gt, gt_lnd)
            if ind is None:
                print(f"skipped {img_path} because of no matching ground truth")
                continue
            reg_targets = compute_reg_targets(anchors, gt, cls, ind)
            lnd_targets, lnd_mask = compute_lnd_targets(anchors, gt_lnd, cls, ind)
            cls = tf.cast(cls, tf.int8)
            lnd_mask = tf.cast(lnd_mask, tf.int8)



            if tf.reduce_any(tf.logical_or(tf.math.is_inf(reg_targets), tf.math.is_nan(reg_targets))):
                print(f"skipped {img_path} because of invalid regression targets")
                continue


            with open(img_path, "rb") as f:
                img_bytes = f.read()
            example = tf.train.Example(features=tf.train.Features(feature={
                "image_raw": _bytes_feature(img_bytes),
                "cls": _int64_feature(cls.numpy().tolist()),
                "reg_targets": _float_feature(tf.reshape(reg_targets, [-1]).numpy().tolist()),
                "lnd_targets": _float_feature(tf.reshape(lnd_targets, [-1]).numpy().tolist()),
                "lnd_mask": _int64_feature(lnd_mask.numpy().tolist())
            }))
            writer.write(example.SerializeToString())
            print(f"added image to dataset:{path}")
    print(f"saved TFRecord file to {output_path}")


def parse_example(example_proto):
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'cls': tf.io.FixedLenFeature([16800], tf.int64),
        'reg_targets': tf.io.FixedLenFeature([16800 * 4], tf.float32),
        'lnd_targets': tf.io.FixedLenFeature([16800 * 10], tf.float32),
        'lnd_mask': tf.io.FixedLenFeature([16800], tf.int64)
    }

    example = tf.io.parse_single_example(example_proto, feature_description)

    image = tf.io.decode_jpeg(example['image_raw'], channels=3)
    cls = tf.cast(example['cls'], tf.int32)
    lnd_mask = tf.cast(example['lnd_mask'], tf.int32)
    reg_targets = example['reg_targets']
    lnd_targets = example['lnd_targets']

    reg_targets = tf.reshape(reg_targets, [16800, 4])
    lnd_targets = tf.reshape(lnd_targets, [16800, 10])

    return image, cls, reg_targets, lnd_targets, lnd_mask

def preprocess_image(image, cls, reg_targets, lnd_targets, lnd_mask):
    image = tf.image.resize(image, (640, 640))
    image = tf.subtract(image, 127.5)
    image = tf.multiply(image, 0.0078125)
    return image, cls, reg_targets, lnd_targets, lnd_mask


def load_dataset(tfrecord_path, batch_size=None, shuffle_buffer=None):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(parse_example)
    dataset = parsed_dataset.map(preprocess_image)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    if shuffle_buffer is not None:
        dataset = dataset.shuffle(shuffle_buffer)
    return dataset


def main():
    anchor_generator = AnchorBoxes(box_sizes=[[16, 32], [64, 128], [256, 512]], image_size=(640, 640),
                                   steps=[8, 16, 32])
    boxes = anchor_generator.generate_anchors()
    boxes = anchor_generator.normalize_anchors(boxes)
    print(boxes.shape)

    lab = get_labels("../Wider_Face_Labels", "../WIDER_train/images")
    lab = lab
    create_tfrecord("../WIDER_train/images", "../WIDER_train/train.tfrecord", lab, boxes)



if __name__ == "__main__":
    main()
