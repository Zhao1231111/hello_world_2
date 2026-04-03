import os
import re
from pathlib import Path

def process_gaussian_logs(directory_path):
    # 目标文件名
    target_filename = "gaussian_lic_log.txt"
    
    # 编译正则表达式，匹配 "[MLP] Regressed xxx Gaussians." 并提取其中的数字
    # \s+ 表示匹配一个或多个空格， \d+ 表示匹配一个或多个数字
    pattern = re.compile(r"\[MLP\]\s+Regressed\s+(\d+)\s+Gaussians\.")
    
    # 追加的文本前缀，也用来作为防重复检查的标识
    append_prefix = "mlp total regressed"
    
    # 获取目录的 Path 对象
    root_dir = Path(directory_path)
    
    # rglob('*') 结合 target_filename 实现递归查找所有同名文件
    target_files = list(root_dir.rglob(target_filename))
    
    if not target_files:
        print(f"在目录 '{directory_path}' 下没有找到 '{target_filename}' 文件。")
        return

    for file_path in target_files:
        total_gaussians = 0
        already_processed = False
        
        try:
            # 1. 读取文件内容
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            # 2. 逐行匹配并累加数字
            for line in lines:
                # 检查文件是否已经被处理过（防重复追加）
                if line.strip().startswith(append_prefix):
                    already_processed = True
                    break
                
                # 正则匹配
                match = pattern.search(line)
                if match:
                    # match.group(1) 提取的就是正则表达式中 (\d+) 匹配到的纯数字
                    num = int(match.group(1))
                    total_gaussians += num
            
            # 3. 追加结果到文件末尾
            if not already_processed:
                with open(file_path, 'a', encoding='utf-8') as f:
                    # 确保追加时是在新的一行
                    if lines and not lines[-1].endswith('\n'):
                        f.write('\n')
                        
                    result_line = f"{append_prefix} {total_gaussians} Gaussians\n"
                    f.write(result_line)
                    
                print(f"✅ 处理成功: {file_path} | 累加总计: {total_gaussians}")
            else:
                print(f"⏩ 跳过 (已处理过): {file_path}")
                
        except Exception as e:
            print(f"❌ 处理文件 {file_path} 时出错: {e}")

if __name__ == "__main__":
    print("="*50)
    print("Gaussian Log 提取累加工具")
    print("="*50)
    
    # 提示用户输入目标目录
    target_dir = input("请输入要扫描的目录绝对路径或相对路径: ").strip()
    
    # 兼容拖拽路径时可能自带的引号
    target_dir = target_dir.strip('\"').strip('\'')
    
    if os.path.exists(target_dir):
        process_gaussian_logs(target_dir)
        print("\n🎉 所有任务执行完毕！")
    else:
        print("\n⚠️ 错误：输入的目录不存在，请检查后重试。")