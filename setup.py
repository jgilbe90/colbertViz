from setuptools import setup, find_packages

setup(
    name='colbertViz',
    version='0.1',
    description='simple library for visualizing document term relevancy in colbert scores',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Joshua Gilbert',
    author_email='joshuag9039@gmail.com',
    packages=find_packages(),
    install_requires=[
       'colbert-ai[torch]',
       'html2image',
       'pyarrow<15'
    ],
)