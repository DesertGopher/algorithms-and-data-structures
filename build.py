import PyInstaller.__main__
import shutil
import os
from cryptography.fernet import Fernet


def build_exe():
    main_file = "main.py"
    directory = "Алгоритмы и структуры данных"

    if os.path.exists(directory):
        shutil.rmtree(directory)

    options = [
        '--onefile',
        '--name', "Алгоритмы и структуры данных",
        '--noconsole',
        '--icon', 'main_window\icon.png',
        '--add-data', 'main_window;main_window',

        '--hidden-import', 'matplotlib',
        '--hidden-import', 'PyQt5',
    ]

    PyInstaller.__main__.run([main_file] + options)

    if os.path.exists('dist'):
        os.rename('dist', f'{directory}')

    if os.path.exists('build'):
        shutil.rmtree('build')

    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".spec"):
            file_path = os.path.join(os.getcwd(), filename)
            os.remove(file_path)


if __name__ == "__main__":
    build_exe()
