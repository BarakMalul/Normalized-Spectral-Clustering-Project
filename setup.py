from setuptools import setup, find_packages, Extension
setup(
    name='spkmeansmodule',
    version='0.1',
    packages=find_packages(),

    ext_modules=[
        Extension(
            "spkmeansmodule",
            ["spkmeansmodule.c", "spkmeans.c"]
        )
    ]
)