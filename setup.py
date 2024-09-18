from setuptools import setup, find_packages
import versioneer

setup(
    name="imshow_mosaic",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["rectpack", "matplotlib", "scikit-image"],
)
