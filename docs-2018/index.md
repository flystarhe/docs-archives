# Home

Welcome to [Docs-2018](#) docs.

## 编译
```bash
mkdocs build --clean
```

## 提交
```bash
git add .
git commit -m "mkdocs build --clean"
git push origin master
```

## 忽略规则
```text
!/docs/**/*
/.*/
/**/tmps/
/**/.cache/
/**/__pycache__/
/**/.ipynb_checkpoints/
/**/.DS_Store
/docs_md/**/*.html
```
