#!/usr/bin/env python3

import numpy as np
import cv2
import math
import time
from typing import Tuple, Union
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        """
        """
        self.MARGIN = 10  # pixels
        self.ROW_SIZE = 10  # pixels
        self.FONT_SIZE = 1
        self.FONT_THICKNESS = 1
        self.TEXT_COLOR = (255, 0, 0)  # red
        self.annotated_image = None

    def _normalized_to_pixel_coordinates(
        self,
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                            math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px


    def visualize(
        self,
        image,
        detection_result
    ) -> np.ndarray:
        """Draws bounding boxes and keypoints on the input image and return it.
        Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualize.
        Returns:
        Image with bounding boxes.
        """
        annotated_image = image.copy()
        height, width, _ = image.shape

        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, self.TEXT_COLOR, 3)

            # Draw keypoints
            for keypoint in detection.keypoints:
                keypoint_px = self._normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                                width, height)
                color, thickness, radius = (0, 255, 0), 2, 2
                cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            category_name = '' if category_name is None else category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (self.MARGIN + bbox.origin_x,
                                self.MARGIN + self.ROW_SIZE + bbox.origin_y)
            cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        self.FONT_SIZE, self.TEXT_COLOR, self.FONT_THICKNESS)

        return annotated_image

    def callback(self, result, output_image, timestamp_ms):
        """
        """
        self.annotated_image = self.visualize(self.frame, result)
        
    def detect_face(self):
        """
        """  
        # STEP 2: Create an FaceDetector object.
        model_asset_path_='models/detector.tflite'

        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        # FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path_),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.callback)
    
        cap = cv2.VideoCapture(0)

        with FaceDetector.create_from_options(options) as detector:
            while True:
                ret, self.frame = cap.read()
                if(ret):
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.frame)
                    
                    frame_timestamp_ms = int(time.time() * 1000) 
                    detector.detect_async(mp_image, frame_timestamp_ms)

                    if(self.annotated_image is not None):
                        cv2.namedWindow("out", cv2.WINDOW_NORMAL)
                        cv2.imshow("out", self.annotated_image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            break

if __name__=="__main__":
    try:
        FD_ = FaceDetector()
        FD_.detect_face()
    except Exception as e:
        print(e)
