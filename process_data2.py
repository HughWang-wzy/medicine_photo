import os
import hashlib
from collections import defaultdict

def calculate_md5(file_path):
    """计算文件的 MD5 哈希值"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def find_duplicates(directory):
    """查找目录中的重复文件"""
    file_hashes = defaultdict(list)
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_hash = calculate_md5(file_path)
            file_hashes[file_hash].append(file_path)
    
    duplicates = {hash: paths for hash, paths in file_hashes.items() if len(paths) > 1}
    return duplicates

def delete_duplicates(duplicates):
    """删除重复文件，只保留 test 目录中的文件"""
    for file_hash, paths in duplicates.items():
        # 按路径排序，确保 test 目录优先
        paths.sort(key=lambda x: 'test' not in x)
        # 保留 test 目录中的文件，删除其他文件
        for path in paths[1:]:  # 从第 2 个文件开始删除
            if 'test' not in path:
                print(f"删除文件: {path}")
                os.remove(path)

# 检查 ISIC_DATA 目录
duplicates = find_duplicates('ISIC_DATA')

if duplicates:
    print("发现重复文件:")
    total_duplicates = 0
    for file_hash, paths in duplicates.items():
        print(f"哈希值: {file_hash}")
        for path in paths:
            print(f"  {path}")
        total_duplicates += len(paths) - 1  # 每组重复文件中，重复的文件数是 len(paths) - 1
    print(f"\n总共发现 {total_duplicates} 个重复文件。")

    # 删除重复文件
    delete_duplicates(duplicates)
    print("\n重复文件已删除，只保留 test 目录中的文件。")
else:
    print("未发现重复文件。")