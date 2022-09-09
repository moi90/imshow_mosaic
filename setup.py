from setuptools import setup
import versioneer

setup(
    name="imshow_mosaic",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=["imshow_mosaic"],
    install_requires=["rectpack", "matplotlib", "scikit-image"],
)
