import setuptools

IMAGE_DEPS = ['Pillow', 'opencv-python']
ML_DEPS = ['torch', 'torchvision']
CLIP = 'clip @ git+https://github.com/openai/CLIP.git@main#egg=clip'
deps = {
    'image': IMAGE_DEPS,
    'audio': ['sounddevice', 'soundfile'],
    'clip': IMAGE_DEPS+['torch', CLIP],
    'yolo': IMAGE_DEPS+['torch'],
    'omnivore': ['hydra-core', 'einops', 'iopath', 'timm'],
    'egovlp': ['transformers', 'av', 'decord', 'ffmpeg', 'humanize', 'psutil', 'transformers', 'timm', 'einops'],
    'detic': ML_DEPS+[CLIP, 'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@main#egg=detectron2', ],
    'egohos': ML_DEPS+['mmcv-full==1.6.0', 'mmsegmentation @ git+https://github.com/owenzlz/EgoHOS.git#egg=mmsegmentation&subdirectory=mmsegmentation'],
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
        'Pillow', 'opencv-python',
        #'ptgctl @ git+ssh://git@github.com/VIDA-NYU/ptgctl@main#egg=ptgctl', 
    ],
    extras_require={
        'test': ['pytest', 'pytest-cov'],
        'doc': ['sphinx-rtd-theme'],
        **deps,
        'all': [vi for v in deps.values() for vi in v],
        'current': [vi for k in ['image', 'egovlp', 'detic', 'egohos'] for vi in deps[k]],
    },
    license='MIT License',
    keywords='ptg machine learning processing recording server local data streams')
