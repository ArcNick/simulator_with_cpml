import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def visualize_snapshots():
    # 从params.txt读取实际尺寸
    params_file = Path("models/params.txt")
    if params_file.exists():
        with open(params_file, 'r') as f:
            lines = f.readlines()
            nx = int([line.split('=')[1].strip() for line in lines if 'nx =' in line][0])
            nz = int([line.split('=')[1].strip() for line in lines if 'nz =' in line][0])
    else:
        # 默认值
        nx, nz = 600, 600
    
    print(f"使用网格尺寸: nx={nx}, nz={nz}")
    
    # 网格尺寸字典 - 使用实际参数
    grid_sizes = {
        "sx": (nz, nx),      # (nz, nx)
        "sz": (nz, nx),      # (nz, nx)
        "txz": (nz - 1, nx - 1),  # txz是(nx-1, nz-1)
        "vx": (nz, nx - 1),   # vx是(nx-1, nz)
        "vz": (nz - 1, nx)    # vz是(nx, nz-1)
    }
    
    base_dir = Path("output")
    output_dir = Path("snapshot_images")
    
    # 确保输出目录存在
    for component in grid_sizes.keys():
        (output_dir / component).mkdir(parents=True, exist_ok=True)
    
    
    vmin, vmax = -0.06, 0.06
    
    # 处理每个分量
    for component, (expected_nz, expected_nx) in grid_sizes.items():
        print(f"处理分量: {component}")
        
        # 源目录
        source_dir = base_dir / component
        
        if not source_dir.exists():
            print(f"警告: 目录 {source_dir} 不存在，跳过")
            continue
        
        # 获取所有二进制文件
        bin_files = sorted(source_dir.glob("*.bin"))
        
        if not bin_files:
            print(f"警告: 在 {source_dir} 中没有找到.bin文件")
            continue
        
        print(f"找到 {len(bin_files)} 个快照文件")
        
        # 处理每个快照文件
        for bin_file in bin_files:
            print(f"处理: {bin_file.name}")
            
            try:
                # 读取二进制文件
                data = np.fromfile(bin_file, dtype=np.float32)
                if len(data) != expected_nz * expected_nx:
                    print(f"警告: 数据大小不匹配，期望 {expected_nz * expected_nx}，实际 {len(data)}，跳过")
                    continue
                    
                data = data.reshape(expected_nz, expected_nx)
                
                # 创建固定尺寸的图形
                fig = plt.figure(figsize=(15, 10), dpi=150)
                
                # 显示图像，使用统一的颜色范围
                im = plt.imshow(data, cmap='seismic', aspect='auto', 
                               vmin=-0.06, vmax=0.06,
                               extent=[0, expected_nx, expected_nz, 0])
                
                # 添加颜色条
                cbar = plt.colorbar(im, shrink=0.8)
                cbar.set_label('Amplitude', rotation=270, labelpad=15)
                
                # 设置标题和标签
                title = f"{component} - {bin_file.stem}"
                plt.title(title, fontsize=14, pad=20)
                plt.xlabel('X Grid Points', fontsize=12)
                plt.ylabel('Z Grid Points', fontsize=12)
                
                # 保存为jpg，使用统一的bbox_inches设置
                output_file = output_dir / component / f"{bin_file.stem}.jpg"
                plt.savefig(output_file, dpi=150, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', pad_inches=0.1)
                plt.close(fig)  # 明确关闭图形
                
                print(f"已保存: {output_file}")
                
            except Exception as e:
                print(f"错误处理文件 {bin_file}: {e}")
    
    print("所有快照处理完成！")

if __name__ == "__main__":
    visualize_snapshots()