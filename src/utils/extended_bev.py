#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import qos_profile_sensor_data
import cv2
import numpy as np
import argparse
import sys


class BEVYAutoSetupNode(Node):
    def __init__(self, args):
        super().__init__('bev_y_auto_setup_node')
        self.args = args
        self.src_points = []
        self.max_points = 4

        self.dst_points = np.float32([
            [0, args.warp_height],
            [args.warp_width, args.warp_height],
            [0, 0],
            [args.warp_width, 0]
        ])

        # 캔버스 관련
        self.canvas_width = args.canvas_width
        self.canvas_height = args.canvas_height
        self.offset_x = 0
        self.offset_y = 0
        self.frame_width = 0
        self.frame_height = 0

        if 'compressed' in args.topic:
            self.msg_type = CompressedImage
        else:
            self.msg_type = Image

        self.subscription = self.create_subscription(
            self.msg_type,
            args.topic,
            self.image_callback,
            qos_profile_sensor_data
        )
        self.get_logger().info(f"Subscribing to {args.topic}...")

        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Original", self.mouse_callback)

        print("\n[사용 방법 - Y축 자동 정렬 모드 (ROS 2 Topic)]")
        print("1. 클릭 순서: 좌하 -> 우하(y고정) -> 좌상 -> 우상(y고정)")
        print("2. 원본 영상은 큰 캔버스 위에 표시됩니다.")
        print("3. 이제 원본 영상 바깥 여백도 클릭 가능합니다.")
        print("4. 저장되는 좌표는 '원본 영상 기준 좌표'입니다.")
        print("   (즉, 여백을 클릭하면 음수 또는 원본 크기보다 큰 좌표가 저장될 수 있습니다.)")
        print("5. 's' 키: 저장 후 종료 / 'r' 키: 마지막 점 취소 / 'q' 키: 종료\n")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.src_points) >= self.max_points:
                print("[WARNING] 이미 4개의 점이 선택되었습니다. 'r'을 눌러 마지막 점을 취소하세요.")
                return

            # 캔버스 좌표 -> 원본 영상 좌표
            img_x = x - self.offset_x
            img_y = y - self.offset_y

            point_order = ["Left-Bottom", "Right-Bottom", "Left-Top", "Right-Top"]
            current_point_index = len(self.src_points)
            final_point = (img_x, img_y)

            # 우측 점(Index 1, 3) 클릭 시 y좌표 자동 정렬
            if current_point_index == 1:   # Right-Bottom
                y_bottom = self.src_points[0][1]
                final_point = (img_x, y_bottom)
            elif current_point_index == 3: # Right-Top
                y_top = self.src_points[2][1]
                final_point = (img_x, y_top)

            self.src_points.append(final_point)

            # 정보 출력
            outside_msg = ""
            if not (0 <= final_point[0] < self.frame_width and 0 <= final_point[1] < self.frame_height):
                outside_msg = " [OUTSIDE original image]"

            print(f"[INFO] Added {point_order[current_point_index]} point: {final_point} ({len(self.src_points)}/{self.max_points}){outside_msg}")

            if len(self.src_points) == self.max_points:
                print("[INFO] 모든 점 선택 완료. 's'를 눌러 저장하거나 'r'로 마지막 점을 취소하세요.")

    def image_callback(self, msg):
        try:
            # Manual Decode
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)

            if self.msg_type == Image:
                if np_arr.size == (msg.width * msg.height * 2):
                    frame = cv2.cvtColor(
                        np_arr.reshape((msg.height, msg.width, 2)),
                        cv2.COLOR_YUV2BGR_YUYV
                    )
                elif np_arr.size == (msg.width * msg.height * 3):
                    frame = np_arr.reshape((msg.height, msg.width, 3))
                    if 'rgb' in msg.encoding.lower():
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                return

            # 원본 프레임 크기 저장
            frame_h, frame_w = frame.shape[:2]
            self.frame_width = frame_w
            self.frame_height = frame_h

            # 캔버스가 원본보다 너무 작으면 자동 보정
            canvas_w = max(self.canvas_width, frame_w)
            canvas_h = max(self.canvas_height, frame_h)

            # 큰 캔버스 생성
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            # 원본 영상을 캔버스 중앙에 배치
            self.offset_x = (canvas_w - frame_w) // 2
            self.offset_y = (canvas_h - frame_h) // 2

            canvas[self.offset_y:self.offset_y + frame_h,
                   self.offset_x:self.offset_x + frame_w] = frame

            disp = canvas.copy()

            # 원본 영상 경계 표시
            cv2.rectangle(
                disp,
                (self.offset_x, self.offset_y),
                (self.offset_x + frame_w - 1, self.offset_y + frame_h - 1),
                (255, 255, 0),
                2
            )

            point_labels = ["1 (L-Bot)", "2 (R-Bot)", "3 (L-Top)", "4 (R-Top)"]

            # 저장된 원본 좌표를 캔버스 좌표로 바꿔서 표시
            for i, pt in enumerate(self.src_points):
                draw_pt = (pt[0] + self.offset_x, pt[1] + self.offset_y)

                # 원본 밖 점이면 색 다르게
                is_inside = (0 <= pt[0] < frame_w and 0 <= pt[1] < frame_h)
                color = (0, 255, 0) if is_inside else (0, 165, 255)  # inside=green, outside=orange

                cv2.circle(disp, draw_pt, 6, color, -1)
                cv2.putText(
                    disp,
                    point_labels[i],
                    (draw_pt[0] + 5, draw_pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            if len(self.src_points) == 4:
                # 화면 표시용 폴리라인은 캔버스 좌표 기준
                draw_pts = np.array(
                    [[pt[0] + self.offset_x, pt[1] + self.offset_y] for pt in self.src_points],
                    dtype=np.int32
                )
                cv2.polylines(disp, [draw_pts], True, (0, 0, 255), 2)

                # 실제 BEV 계산은 원본 영상 좌표 기준
                try:
                    M = cv2.getPerspectiveTransform(np.float32(self.src_points), self.dst_points)
                    bev_result = cv2.warpPerspective(
                        frame,
                        M,
                        (self.args.warp_width, self.args.warp_height)
                    )
                    cv2.imshow("BEV", bev_result)
                except cv2.error as e:
                    self.get_logger().warn(f"Perspective transform failed: {e}")

            # 안내 문구
            cv2.putText(
                disp,
                "Cyan box = original image area",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )
            cv2.putText(
                disp,
                "Green: inside / Orange: outside original image",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2
            )
            cv2.putText(
                disp,
                "Outside clicks are now allowed",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2
            )

            cv2.imshow("Original", disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                rclpy.shutdown()

            elif key == ord('r'):
                if len(self.src_points) > 0:
                    self.src_points.pop()
                    print(f"[INFO] 마지막 점이 취소되었습니다. 현재 선택된 점 수: {len(self.src_points)}/{self.max_points}")
                else:
                    print("[INFO] 취소할 점이 없습니다.")

            elif key == ord('s'):
                if len(self.src_points) < 4:
                    print("[WARNING] 4개의 점을 모두 선택해야 저장 가능합니다.")
                else:
                    np.savez(
                        self.args.out_npz,
                        src_points=np.float32(self.src_points),
                        dst_points=self.dst_points,
                        warp_w=self.args.warp_width,
                        warp_h=self.args.warp_height
                    )

                    try:
                        with open(self.args.out_txt, 'w') as f:
                            f.write("# Selected BEV Points (Y-Auto, original image coordinates)\n")
                            f.write("# Points can be outside original image bounds\n")
                            f.write(f"# Original image size: {frame_w} x {frame_h}\n")
                            f.write(f"# Canvas size: {canvas_w} x {canvas_h}\n")
                            f.write(f"# Offset: ({self.offset_x}, {self.offset_y})\n")
                            for i, pt in enumerate(self.src_points):
                                outside = not (0 <= pt[0] < frame_w and 0 <= pt[1] < frame_h)
                                mark = "OUTSIDE" if outside else "INSIDE"
                                f.write(f"{pt[0]}, {pt[1]} # {point_labels[i]} [{mark}]\n")

                        print(f"[INFO] 저장 완료: {self.args.out_npz}, {self.args.out_txt}")

                    except Exception as e:
                        print(f"[ERROR] 파일 저장 중 오류 발생: {e}")

                    rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Image Callback Error: {e}")


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default='/image_raw/compressed',
                        help='구독할 ROS 2 이미지 토픽 이름')
    parser.add_argument('--warp-width', type=int, default=640)
    parser.add_argument('--warp-height', type=int, default=640)
    parser.add_argument('--canvas-width', type=int, default=1000,
                        help='표시용 캔버스 너비')
    parser.add_argument('--canvas-height', type=int, default=800,
                        help='표시용 캔버스 높이')
    parser.add_argument('--out-npz', type=str, default='bev_params_7.npz')
    parser.add_argument('--out-txt', type=str, default='selected_bev_src_points_7.txt')

    args = parser.parse_args()

    node = BEVYAutoSetupNode(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()