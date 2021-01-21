from setuptools import setup, find_packages, find_namespace_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='cania-core',
      version='0.1',
      description='Cancer Image Analysis python package',
      long_description='',
      classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Healthcare Industry',
      ],
      keywords='cancer artificial-intelligence computer vision',
      url='https://github.com/Cancer-Image-Analysis/cania-core',
      author='Kevin Cortacero',
      author_email='kevin.cortacero@inserm.fr',
      license='MIT',
      packages=find_namespace_packages(where='src'),
      package_dir={'cania': 'src/cania'},
      install_requires=[
          'tifffile',
          'openslide-python',
          'pandas'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
      