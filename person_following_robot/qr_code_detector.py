#!/usr/bin/env python3

from rclpy import Node

class QrCodeDetector(Node):
    def __init__(self) -> None:
        super().__init__('qr_code_detector')
