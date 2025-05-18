#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import json
import time
import cv2
import argparse
import numpy as np
import sophon.sail as sail
import logging
import threading
import queue
import base64
import sys

from utils import *
logging.basicConfig(level=logging.INFO)

# RTSP Stream Handling Globals
rtsp_frame_queue = queue.Queue(maxsize=5)
rtsp_result_queue = queue.Queue(maxsize=5)
rtsp_thread_running = False
rtsp_thread = None
rtsp_url = "rtsp://127.0.0.1:8554/live"  # Default RTSP URL

class YOLOv8:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
              
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        self.agnostic = False
        self.multi_label = False
        self.max_det = 300

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess(self, ori_img):
        """
        pre-processing
        Args:
            img: numpy.ndarray -- (h,w,3)

        Returns: (3,h,w) numpy.ndarray after pre-processing

        """
        letterbox_img, ratio, (tx1, ty1) = self.letterbox(
            ori_img,
            new_shape=(self.net_h, self.net_w),
            color=(114, 114, 114),
            auto=False,
            scaleFill=False,
            scaleup=True,
            stride=32
        )

        img = letterbox_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img.astype(np.float32)
        # input_data = np.expand_dims(input_data, 0)
        img = np.ascontiguousarray(img / 255.0)
        return img, ratio, (tx1, ty1) 

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return im, ratio, (dw, dh)

    def predict(self, input_img, img_num):
        input_data = {self.input_name: input_img}
        outputs = self.net.process(self.graph_name, input_data)
        
        # resort
        out_keys = list(outputs.keys())
        ord = []
        for n in self.output_names:
            for i, k in enumerate(out_keys):
                if n == k:
                    ord.append(i)
                    break
        out = [outputs[out_keys[i]][:img_num] for i in ord]
        return out
    
    def postprocess(self, preds, ori_size_list, ratio_list, txy_list):
        """
        post-processing
        Args:
            preds: numpy.ndarray -- (n,8400,56) [cx,cy,w,h,conf,17*3]

        Returns: 
            results: list of numpy.ndarray -- (n, 56) [x1, y1, x2, y2, conf, 17*3]

        """
        results = []
        preds = preds[0]
        for i, pred in enumerate(preds):
            # Transpose and squeeze the output to match the expected shape
            pred = np.transpose(pred, (1, 0))   # [8400,56]

            pred = pred[pred[:, 4] > self.conf_thresh]

            if len(pred) == 0:
                print("none detected")
                results.append(np.zeros((0, 56)))
            else:
                # Box (center x, center y, width, height) to (x1, y1, x2, y2)
                pred = self.xywh2xyxy(pred)
                results.append(self.nms_boxes(pred, self.nms_thresh))


        # Rescale boxes and keypoints from img_size to im0 size
        for det, (org_w, org_h), ratio, (tx1, ty1) in zip(results, ori_size_list, ratio_list, txy_list):
            if len(det):
                # Rescale boxes from img_size to im0 size
                coords = det[:, :4]
                coords[:, [0, 2]] -= tx1  # x padding
                coords[:, [1, 3]] -= ty1  # y padding
                coords[:, [0, 2]] /= ratio[0]
                coords[:, [1, 3]] /= ratio[1]

                coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, org_w - 1)  # x1, x2
                coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, org_h - 1)  # y1, y2

                det[:, :4] = coords

                # Rescale keypoints from img_size to im0 size
                num_kpts = (det.shape[1] - 5) // 3
                for kid in range(num_kpts):
                    det[:, 5 + kid * 3] -= tx1
                    det[:, 5 + kid * 3 + 1] -= ty1
                    det[:, 5 + kid * 3] /= ratio[0]
                    det[:, 5 + kid * 3 + 1] /= ratio[1]
                    det[:, 5 + kid * 3] = det[:, 5 + kid * 3].clip(0, org_w - 1)
                    det[:, 5 + kid * 3 + 1] = det[:, 5 + kid * 3 + 1].clip(0, org_h - 1)

        return results

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.copy() if isinstance(x, np.ndarray) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def nms_boxes(self, pred, iou_thres):
        x = pred[:, 0]
        y = pred[:, 1]
        w = pred[:, 2] - pred[:, 0]
        h = pred[:, 3] - pred[:, 1]

        scores = pred[:, 4]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]
        output = []
        for i in keep:
            output.append(pred[i].tolist())
        return np.array(output)

    def __call__(self, img_list):
        img_num = len(img_list)
        ori_size_list = []
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            start_time = time.time()
            preprocessed_img, ratio, (tx1, ty1) = self.preprocess(ori_img)
            self.preprocess_time += time.time() - start_time
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)
            txy_list.append([tx1, ty1])
        
        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)
        
        start_time = time.time()
        outputs = self.predict(input_img, img_num)

        self.inference_time += time.time() - start_time

        start_time = time.time()
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)

        self.postprocess_time += time.time() - start_time

        return results


def draw_numpy(image, det_draw, masks=None, classes_ids=None, conf_scores=None):
    boxes = det_draw[:, :4]
    kpts = det_draw[:, 5:]
    num_kpts = (kpts.shape[1]) // 3
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        if classes_ids is not None:
            color = COLORS[int(classes_ids[idx]) + 1]
        else:
            color = (0, 0, 255)
        # draw boxs
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        if classes_ids is not None and conf_scores is not None:
            classes_ids = classes_ids.astype(np.int8)
            cv2.putText(image, COCO_CLASSES[classes_ids[idx] + 1] + ':' + str(round(conf_scores[idx], 2)),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)
        if masks is not None:
            mask = masks[:, :, idx]
            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5
        
        # draw keypoints
        for i in range(0, len(kpts[idx]), 3):
            x, y, conf = kpts[idx, i:i + 3]
            if conf > 0.5:
                cv2.circle(image, (int(x), int(y)), 5, color, -1)
        
        # draw skeleton
        for sk_id, sk in enumerate(skeleton):
            r, g, b = pose_limb_color[sk_id]
            pos1 = (int(kpts[idx, (sk[0]-1)*3]), int(kpts[idx, (sk[0]-1)*3+1]))
            pos2 = (int(kpts[idx, (sk[1]-1)*3]), int(kpts[idx, (sk[1]-1)*3+1]))
            conf1 = kpts[idx, (sk[0]-1)*3+2]
            conf2 = kpts[idx, (sk[1]-1)*3+2]
            if conf1 >0.5 and conf2 >0.5:
                cv2.line(image, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

        
        logging.debug("score={}, (x1={},y1={},x2={},y2={})".format(conf_scores[idx], x1, y1, x2, y2))
    return image

def rtsp_detection_thread(yolov8, args):
    """
    Background thread that continuously reads from RTSP stream and performs detection
    """
    global rtsp_thread_running
    global rtsp_url
    
    print(f"RTSP detection thread starting, connecting to: {rtsp_url}")
    
    try:
        # Check if RTSP URL is valid
        if not rtsp_url or not rtsp_url.startswith('rtsp://'):
            print(f"Error: Invalid RTSP URL: {rtsp_url}")
            print("RTSP URL must start with 'rtsp://'")
            rtsp_thread_running = False
            return
            
        print(f"Attempting to connect to RTSP stream: {rtsp_url}")
        
        # Setup RTSP connection
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;15000000"  # 15 sec timeout
        
        # Try to open RTSP stream
        print("Creating VideoCapture object...")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        print("Setting buffer size...")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        # Set read timeout
        if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT'):
            print("Setting connection timeout to 15 seconds...")
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT, 15000)  # 15 sec connection timeout
        if hasattr(cv2, 'CAP_PROP_READ_TIMEOUT'):
            print("Setting read timeout to 5 seconds...")
            cap.set(cv2.CAP_PROP_READ_TIMEOUT, 5000)   # 5 sec read timeout
        
        if not cap.isOpened():
            print(f"Error: Unable to connect to RTSP stream {rtsp_url}")
            print("Please check: 1) Device is on 2) IP address is correct 3) Port is correct 4) Path is correct")
            rtsp_thread_running = False
            return
        
        print("RTSP detection thread started successfully!")
        
        frame_count = 0
        while rtsp_thread_running:
            # Read frame
            ret, frame = cap.read()
            
            # If read failed
            if not ret:
                print("Frame read failed, attempting to reconnect...")
                # Try to close old connection
                cap.release()
                # Wait a bit
                time.sleep(1)
                # Reconnect
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, cv2.CAP_RTSP_TCP)
                continue
            
            # Process the frame with YOLOv8
            results = yolov8([frame])
            
            if results[0].shape[0] > 0:
                # Draw results on frame
                det = results[0]
                det_draw = det[det[:, 4] > args.conf_thresh]
                annotated_frame = draw_numpy(frame.copy(), det_draw, masks=None, classes_ids=None, conf_scores=det_draw[:, 4])
                
                # Update frame queue
                if rtsp_frame_queue.full():
                    rtsp_frame_queue.get()
                rtsp_frame_queue.put(frame)
                
                # Extract keypoints for further processing
                kpts = det_draw[:, 5:]
                
                # Create results dictionary
                try:
                    # Convert to JPEG format for result queue
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    
                    # Update result queue
                    if rtsp_result_queue.full():
                        rtsp_result_queue.get()
                    
                    # Create result dictionary with keypoints and image
                    result_dict = {
                        'image': f"data:image/jpeg;base64,{img_str}",
                        'frame_count': frame_count,
                        'timestamp': time.time(),
                        'boxes': det_draw[:, :4].tolist(),
                        'scores': det_draw[:, 4].tolist(),
                        'keypoints': []
                    }
                    
                    # Add keypoints to result
                    for n in range(kpts.shape[0]):
                        person_kpts = []
                        for m in range(0, len(kpts[n]), 3):
                            x, y, score = kpts[n, m:m + 3]
                            person_kpts.append([float(x), float(y), float(score)])
                        result_dict['keypoints'].append(person_kpts)
                    
                    rtsp_result_queue.put(result_dict)
                    
                except Exception as e:
                    print(f"Error processing RTSP frame: {e}")
            else:
                # No detection - still update the frame queue
                if rtsp_frame_queue.full():
                    rtsp_frame_queue.get()
                rtsp_frame_queue.put(frame)
                
                # Also update the result queue
                try:
                    _, buffer = cv2.imencode('.jpg', frame)
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    
                    if rtsp_result_queue.full():
                        rtsp_result_queue.get()
                    
                    rtsp_result_queue.put({
                        'image': f"data:image/jpeg;base64,{img_str}",
                        'frame_count': frame_count,
                        'timestamp': time.time(),
                        'message': "No detections",
                        'boxes': [],
                        'scores': [],
                        'keypoints': []
                    })
                except Exception as e:
                    print(f"Error processing empty RTSP frame: {e}")
            
            frame_count += 1
            
            # Short sleep to avoid maxing out CPU
            time.sleep(0.01)
        
        # Close RTSP connection
        cap.release()
        print("RTSP detection thread stopped")
    
    except Exception as e:
        print(f"RTSP detection thread error: {e}")
        import traceback
        traceback.print_exc()
        rtsp_thread_running = False

def get_latest_result():
    """
    Get the latest detection result from the queue
    """
    if rtsp_result_queue.empty():
        return None
    
    return rtsp_result_queue.queue[0]  # Get but don't remove

def start_rtsp_thread(yolov8, args):
    """
    Start the RTSP detection thread
    """
    global rtsp_thread, rtsp_thread_running, rtsp_url
    
    # If thread already running, return success
    if rtsp_thread_running and rtsp_thread and rtsp_thread.is_alive():
        print("RTSP detection already running")
        return True
    
    # Update RTSP URL (if provided)
    if args.rtsp_url:
        rtsp_url = args.rtsp_url
        print(f"RTSP URL updated to: {rtsp_url}")
    
    # Start thread
    rtsp_thread_running = True
    rtsp_thread = threading.Thread(target=rtsp_detection_thread, args=(yolov8, args))
    rtsp_thread.daemon = True
    rtsp_thread.start()
    
    return True

def stop_rtsp_thread():
    """
    Stop the RTSP detection thread
    """
    global rtsp_thread_running
    
    if rtsp_thread_running:
        rtsp_thread_running = False
        # Wait for thread to end
        if rtsp_thread:
            rtsp_thread.join(timeout=5.0)
        
        # Clear queues
        while not rtsp_frame_queue.empty():
            rtsp_frame_queue.get()
        while not rtsp_result_queue.empty():
            rtsp_result_queue.get()
        
        print("RTSP detection stopped")
        return True
    else:
        print("RTSP detection not running")
        return False

def main(args):
    # check bmodel
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    
    # create output directory
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir) 

    # Initialize YOLOv8 model
    yolov8 = YOLOv8(args)
    batch_size = yolov8.batch_size
    yolov8.init()
    
    # Start RTSP processing instead of local files/videos
    try:
        if start_rtsp_thread(yolov8, args):
            print(f"Started RTSP processing from {rtsp_url}")
            
            # Keep the main thread running to display or save results
            frame_count = 0
            save_interval = args.save_interval  # Save every N frames
            
            try:
                while True:
                    result = get_latest_result()
                    if result:
                        # Display the latest result if display is enabled
                        if args.display:
                            # Decode base64 image
                            img_data = result['image'].split(',')[1]
                            img_bytes = base64.b64decode(img_data)
                            nparr = np.fromstring(img_bytes, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            # Show the image
                            cv2.imshow("RTSP YOLOv8 Detection", img)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                break
                        
                        # Save frame at specified intervals if saving is enabled
                        if args.save and frame_count % save_interval == 0:
                            # Decode base64 image
                            img_data = result['image'].split(',')[1]
                            img_bytes = base64.b64decode(img_data)
                            nparr = np.fromstring(img_bytes, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            # Save image
                            filename = f"frame_{frame_count:06d}.jpg"
                            cv2.imwrite(os.path.join(output_img_dir, filename), img)
                            
                            # Save JSON with detection results
                            json_filename = f"frame_{frame_count:06d}.json"
                            with open(os.path.join(output_dir, json_filename), 'w') as f:
                                # Remove the image data to keep JSON size reasonable
                                result_copy = result.copy()
                                result_copy.pop('image', None)
                                json.dump(result_copy, f, indent=4)
                        
                        frame_count += 1
                    
                    # Sleep to avoid high CPU usage
                    time.sleep(0.01)
                    
            except KeyboardInterrupt:
                print("Interrupted by user")
            finally:
                # Clean up
                if args.display:
                    cv2.destroyAllWindows()
                stop_rtsp_thread()
    
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        stop_rtsp_thread()

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--rtsp_url', type=str, default=None, help='RTSP URL to connect to')
    input_group.add_argument('--input', type=str, default=None, help='RTSP URL to connect to (legacy parameter)')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/yolov8s-pose_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='nms threshold')
    parser.add_argument('--display', action='store_true', help='display detection results')
    parser.add_argument('--save', action='store_true', help='save detection results')
    parser.add_argument('--save_interval', type=int, default=30, help='save every N frames')
    args = parser.parse_args()
    
    # 处理参数兼容性：如果指定了--input而没有指定--rtsp_url，将input的值赋给rtsp_url
    if args.rtsp_url is None and args.input is not None:
        args.rtsp_url = args.input
    
    # 如果两者都未指定，使用默认值
    if args.rtsp_url is None:
        args.rtsp_url = 'rtsp://10.148.165.1:8554/live'
        
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')