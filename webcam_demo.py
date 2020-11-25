import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto(device_count = {'GPU': 0})
config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)


import cv2
import time
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=640)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def main():
    with tf.Session(config=config) as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id, cv2.CAP_DSHOW)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0


        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )
            #print(heatmaps_result[0,:,:,0])
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            heatmap = cv2.resize(heatmaps_result[0,:,:,0], (640,480))
            #print(overlay_image.shape)

            #plt.imshow(heatmaps_result[0,:,:,0], cmap='hot', interpolation='nearest')
            #plt.show()
            #dst = cv2.addWeighted(overlay_image, 1, heatmap, 1, 0)

            #rgb = cv2.cvtColor(heatmap,cv2.COLOR_GRAY2RGB)
            # print(overlay_image.shape)
            #print(np.max(rgb))

            #new_image = cv2.addWeighted(overlay_image.astype("float32"),0.7,rgb,0.3,0)

            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_count % 100 == 0:
                print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()
