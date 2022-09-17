import setuptools

IMAGE_DEPS = ['Pillow', 'opencv-python']
deps = {
    'image': IMAGE_DEPS,
    'audio': ['sounddevice', 'soundfile'],
    'clip': IMAGE_DEPS+[
        'torch', 
        #'clip @ git+ssh://git@github.com/openai/CLIP@main#egg=clip'
        'clip @ git+https://github.com/openai/CLIP.git@main#egg=clip',
    ],
    'yolo': IMAGE_DEPS+['torch'],
    'omnivore': ['hydra-core', 'einops', 'iopath', 'timm'],
    'egovlp': ['transformers'],
}

setuptools.setup(
    name='ptgprocess',
    version='0.0.1',
    description='PTG Data Processing and Machine Learning framework',
    long_description=open('README.md').read().strip(),
    long_description_content_type='text/markdown',
    author='Bea Steers',
    author_email='bea.steers@gmail.com',
    url=f'https://github.com/VIDA-NYU/ptg-server-ml',
    packages=['ptgprocess'],
    # entry_points={'console_scripts': ['ptgprocess=ptgprocess:main']},
    install_requires=[
        'numpy', 'orjson', 'tqdm',
        #'ptgctl @ git+ssh://git@github.com/VIDA-NYU/ptgctl@main#egg=ptgctl', 
    ],
    extras_require={
        'test': ['pytest', 'pytest-cov'],
        'doc': ['sphinx-rtd-theme'],
        **deps,
        'all': {vi for v in deps.values() for vi in v},
    },
    license='MIT License',
    keywords='ptg machine learning processing recording server local data streams')
