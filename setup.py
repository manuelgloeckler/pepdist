from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='pepdist',
      version='0.1',
      description='Distance and Similarity Methods for fast nearest neighbour search.',
	  long_description=readme(),
      url='https://github.com/manuelgloeckler/immu-thesis.git',
      author='Manuel Gloeckler',
      author_email='manuel.gloeckler@student.uni-tuebingen.de',
      license='No',
      packages=['pepdist'],
      install_requires=[
          'numpy',
		  'pandas',
		  'scipy',
          'multiprocess'
      ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
	  include_package_data=True,
      zip_safe=False)