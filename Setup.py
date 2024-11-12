from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove newline characters
        requirements = [req.replace("\n", "") for req in requirements]

        # Handle the '-e .' entry for editable installs
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="Ml_Project",
    version="0.0.1",
    author="Nishanth",
    author_email="nishanthch1627@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
