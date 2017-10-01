from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='mmpy',
    version='0.1',
    description='Scientific python package to experiment with Maths and Music',
    long_description=readme(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Framework :: Jupyter',
        'Intended Audience :: Education',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development',
    ],
    url='http://github.com/jorgesep/mmpy-pkg',
    author='Jorge Sepulveda',
    author_email='jorge_a_sepulveda@yahoo.com',
    license='UCHILE',
    packages=['mmpy'],
    install_requires=[
        'numpy','scipy','seaborn','matplotlib','soundfile','sounddevice'
    ],
    zip_safe=False)
