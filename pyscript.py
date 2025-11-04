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
        nx, nz = 601, 601
    
    print(f"使用网格尺寸: nx={nx}, nz={nz}")
    
    # 计算长宽比
    aspect_ratio = nx / nz
    
    # 网格尺寸字典 - 使用实际参数
    grid_sizes = {
        "sx": (nz, nx),      # (nz, nx)
        "sz": (nz, nx),      # (nz, nx)
        "txz": (nz - 1, nx - 1),  # txz是(nx-1, nz-1)
        "vx": (nz, nx - 1),   # vx是(nx-1, nz)
        "vz": (nz - 1, nx)    # vz是(nx, nz-1)
    }
    
    # 分开设置应力和速度的显示范围
    stress_range = (-0.06, 0.06)      # 应力分量范围
    velocity_range = (-1e-8, 1e-8)    # 速度分量范围
    
    base_dir = Path("output")
    output_dir = Path("snapshot_images")
    
    # 确保输出目录存在
    for component in grid_sizes.keys():
        (output_dir / component).mkdir(parents=True, exist_ok=True)
    
    # 处理每个分量
    for component, (expected_nz, expected_nx) in grid_sizes.items():
        print(f"处理分量: {component}")
        
        # 根据分量类型选择显示范围
        if component in ["sx", "sz", "txz"]:
            vmin, vmax = stress_range
            range_type = "应力"
        else:  # vx, vz
            vmin, vmax = velocity_range
            range_type = "速度"
        
        print(f"  {range_type}分量，显示范围: [{vmin}, {vmax}]")
        
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
                
                # 创建图形，使用固定长宽比
                base_width = 10  # 基础宽度
                fig_height = base_width / aspect_ratio
                
                # 创建图形
                fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=150)
                
                # 显示图像，严格使用手动设置的范围
                im = ax.imshow(data, cmap='seismic', aspect='equal', 
                              vmin=vmin, vmax=vmax,
                              extent=[0, expected_nx, expected_nz, 0])
                
                # 添加颜色条
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Amplitude', rotation=270, labelpad=15)
                
                # 设置标题和标签
                title = f"{component} - {bin_file.stem}"
                ax.set_title(title, fontsize=12, pad=20)
                ax.set_xlabel('X Grid Points', fontsize=10)
                ax.set_ylabel('Z Grid Points', fontsize=10)
                
                # 保存为jpg
                output_file = output_dir / component / f"{bin_file.stem}.jpg"
                plt.savefig(output_file, dpi=150, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', pad_inches=0.1)
                plt.close(fig)
                
                print(f"已保存: {output_file}")
                
            except Exception as e:
                print(f"错误处理文件 {bin_file}: {e}")
    
    print("所有快照处理完成！")

if __name__ == "__main__":
    visualize_snapshots()