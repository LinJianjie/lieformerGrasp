from pathlib import Path


def get_project_root():
    file = Path(__file__).resolve()
    return file.parents[5]


if __name__ == '__main__':
    print(get_project_root())
