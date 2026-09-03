from glob import glob
import os

from setuptools import setup


package_name = 'mbari_wec_oct_controllers_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[f'{package_name}'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='eagan',
    maintainer_email='ajeagan@uw.edu',
    description='MBARI Power Buoy Controllers for October 2026 Deployment',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'oct_controllers = {package_name}.controller:main',
        ],
    },
)
