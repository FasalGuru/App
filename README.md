### BUILD COMMAND

> `
pyinstaller --onefile --windowed --add-data "assets;assets" --add-data "models;models" --hidden-import=torch --hidden-import=torchvision --hidden-import=numpy --icon="Fasal.ico" --name=Fasal main.py
`