#!/bin/bash

# 设置工作目录 - 替换为你的git仓库路径
REPO_DIR="./"

# 进入仓库目录
cd $REPO_DIR || exit 1

# 获取当前时间戳
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# 执行git操作
git add .
git commit -m "Auto commit at $TIMESTAMP"
git push origin main

# 记录日志
echo "Executed git operations at $TIMESTAMP" >> "$REPO_DIR/git-auto-commit.log"
