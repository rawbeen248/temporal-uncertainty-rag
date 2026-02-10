"""
Setup script for Temporal Uncertainty RAG package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='temporal-uncertainty-rag',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Temporal Uncertainty Tracking in Conversational RAG',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/temporal-uncertainty-rag',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'jupyter>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'temporal-rag-train=scripts.train:main',
            'temporal-rag-eval=scripts.evaluate:main',
            'temporal-rag-prepare=scripts.prepare_data:main',
        ],
    },
)
