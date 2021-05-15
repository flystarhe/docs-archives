# ReadMe

## Installation
```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install -U pip
pip install -U mkdocs
pip install -U mkdocs-material
mkdocs --version
```

## Build
```
mkdocs build --clean
```

## Push
```
git add .
git commit -m "docs/*"
git push origin master
```
