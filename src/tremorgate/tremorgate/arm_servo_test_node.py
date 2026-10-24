#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
import serial


class ArmServoTestNode(Node):
    def __init__(self):
        super().__init__('arm_servo_test_node')
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('step_degrees', 20)
        self.declare_parameter('settle_seconds', 1.5)

        self.port = self.get_parameter('port').value
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.step = int(self.get_parameter('step_degrees').value)
        self.settle = float(self.get_parameter('settle_seconds').value)

        self.joints = [
            (1, 'Waist/base'),
            (2, 'Shoulder'),
            (3, 'Elbow'),
            (4, 'Wrist roll'),
            (5, 'Wrist pitch'),
            (6, 'Gripper'),
        ]

    def send_command(self, connection, command, wait=0.4):
        self.get_logger().info(f'Command: {command}')
        connection.write((command + '\n').encode('ascii'))
        connection.flush()
        time.sleep(wait)

        while connection.in_waiting:
            line = connection.readline().decode(
                'utf-8', errors='replace'
            ).strip()
            if line:
                self.get_logger().info(f'Arduino: {line}')

    def run_test(self):
        if self.step != 20:
            raise ValueError('This test is configured for exactly 20 degrees.')

        connection = None

        try:
            connection = serial.Serial(
                self.port, self.baudrate, timeout=0.2, write_timeout=1
            )
            time.sleep(2)
            connection.reset_input_buffer()

            self.send_command(connection, 'HELP')
            self.get_logger().warning(
                'Only one servo will be attached at a time. '
                'Enter a starting angle that matches the joint position '
                'as closely as possible.'
            )

            input('Switch on servo power only when ready; clear the arm and press ENTER, or Ctrl+C to cancel: ')

            for number, name in self.joints:
                raw = input(
                    f'{name}: approximate current angle (70-110), or q to stop: '
                ).strip()

                if raw.lower() == 'q':
                    break

                try:
                    start = int(raw)
                except ValueError:
                    self.get_logger().error('Invalid angle; stopping test.')
                    break

                if not 70 <= start <= 110:
                    self.get_logger().error('Angle must be between 70 and 110; stopping.')
                    break

                # Move away from the starting angle without exceeding 70-110.
                target = start + 20 if start <= 90 else start - 20

                self.get_logger().warning(
                    f'{name}: starting at {start}°, moving to {target}°, '
                    f'then returning to {start}°.'
                )

                try:
                    self.send_command(connection, f'TEST {number} {start}')
                    input(
                        f'Check {name} is clear. Press ENTER to move this joint, '
                        'or Ctrl+C to cancel: '
                    )
                    self.send_command(
                        connection, f'SET {number} {target}', self.settle
                    )
                    self.send_command(
                        connection, f'SET {number} {start}', self.settle
                    )
                finally:
                    self.send_command(connection, 'DISARM', 0.3)

                self.get_logger().info(f'{name} test finished.')

            self.get_logger().info('Individual servo test finished.')

        except KeyboardInterrupt:
            self.get_logger().warning('Test cancelled.')

        finally:
            if connection is not None:
                try:
                    self.send_command(connection, 'STOP', 0.2)
                    self.send_command(connection, 'DISARM', 0.2)
                except Exception as exc:
                    self.get_logger().error(f'Shutdown command failed: {exc}')
                connection.close()

            self.get_logger().info('Serial connection closed.')


def main(args=None):
    rclpy.init(args=args)
    node = ArmServoTestNode()
    try:
        node.run_test()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
