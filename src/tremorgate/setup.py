from setuptools import setup
from glob import glob
import os

package_name = 'tremorgate'
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'serial_bridge_node = tremorgate.serial_bridge_node:main',
            'perception_node = tremorgate.perception_node:main',
            'arm_gate_node = tremorgate.arm_gate_node:main',
            'arm_stub_node = tremorgate.arm_stub_node:main',
            'replay_serial = tremorgate.replay_serial:main',
            'arm_servo_test_node = tremorgate.arm_servo_test_node:main',
        ],
    },
)
