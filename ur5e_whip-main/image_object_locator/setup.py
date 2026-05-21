from setuptools import setup, find_packages

package_name = 'image_object_locator'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    install_requires=['setuptools', 'opencv-python', 'cv-bridge'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    entry_points={
        'console_scripts': [
            'image_subscriber = image_object_locator.image_subscriber:main',
            'fake_target_publisher = image_object_locator.fake_target_publisher:main',
        ],
    },
)
