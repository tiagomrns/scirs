#\!/bin/bash
# 単一のクレートを公開するシンプルなスクリプト

if [ -z "$1" ]; then
    echo "使用法: $0 <crate-directory>"
    exit 1
fi

CRATE=$1
echo "===== $CRATE を公開します ====="
cd "$CRATE" || { echo "ディレクトリが見つかりません: $CRATE"; exit 1; }
cargo publish --allow-dirty || { echo "公開に失敗しました: $CRATE"; exit 1; }
echo "✓ $CRATE を公開しました"
