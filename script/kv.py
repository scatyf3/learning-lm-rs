import re
import numpy as np
import matplotlib.pyplot as plt

def parse_data_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Check if there are enough lines
    if len(lines) < 3:
        raise ValueError("Input file does not contain enough data.")
    
    # Process the first line to get shape, offset, and length
    first_line = lines[2]
    shape_match = re.search(r'shape: \[(\d+), (\d+)\]', first_line)
    offset_match = re.search(r'offset: (\d+)', first_line)
    length_match = re.search(r'length: (\d+)', first_line)
    
    if shape_match and offset_match and length_match:
        # layers = int(shape_match.group(1))  # Total layers
        shape = (int(shape_match.group(1)), int(shape_match.group(2)))
        offset = int(offset_match.group(1))
        length = int(length_match.group(1))
        
        # Initialize a dictionary to hold data for each layer
        data_dict = {}

        # Extract array data for each layer
        for layer in range(4):
            data_list = []
            # Each layer has shape[1] lines of data, each separated by one line
            print("layer")
            print(layer)
            for i in range(shape[0]):
                data_line_index = 3 + layer * shape[0] + i + layer # add layer offset，避免读入总信息行
                # print(data_line_index)
                data_line = lines[data_line_index]
                data_match = re.search(r'\[(.*?)\]', data_line)
                if data_match:
                    data_str = data_match.group(1)
                    row_data = np.fromstring(data_str, sep=',', dtype=np.float32)
                    if row_data.size != shape[1]:
                        print(row_data)
                        raise ValueError(f"Row data size {row_data.size} does not match expected size {shape[1]}.")
                    data_list.append(row_data)    
                else:
                    print(lines[data_line_index])
                    raise ValueError(f"Data line {data_line_index} is not in the expected format.")

            # Convert the list of arrays into a 2D NumPy array for this layer
            data = np.vstack(data_list)
            
            # Ensure the data has the correct shape
            if data.shape != (shape[0], shape[1]):
                raise ValueError(f"Data shape {data.shape} does not match expected shape {shape}.")
            
            # Store the data for this layer
            data_dict[f'Layer {layer + 1}'] = data

        return shape, offset, length, data_dict
    
    raise ValueError("Could not parse shape, offset, length, or data.")

# Usage example
file_path = 'KVData'  # Replace with your file path

shape,offset,length,data_dict = parse_data_from_file(file_path)

# 删除0
data_dict = {
    layer: data[~np.all(data == 0, axis=1)]
    for layer, data in data_dict.items()
}

print(data_dict)

# 可视化每一层的数据
num_layers = len(data_dict)
plt.figure(figsize=(10, 5 * num_layers))  # 设置图形大小

for i, (layer, data) in enumerate(data_dict.items()):
    plt.subplot(num_layers, 1, i + 1)  # 创建子图
    plt.imshow(data, aspect='auto', cmap='hot', interpolation='nearest')  # 绘制热力图
    plt.colorbar()  # 添加颜色条
    plt.title(f"Heatmap for {layer}")
    plt.xlabel("Embedding Dim")
    plt.ylabel("Seq")

plt.tight_layout()  # 自动调整子图间距
# 保存图像到文件
plt.savefig("heatmaps.png")  # 将图像保存为 PNG 文件
plt.close()  # 关闭图形以释放资源



# Set bins
bins = np.arange(-2, 2.1, 0.1)
distribution_counts = {}

# Count value distributions
for layer, data in data_dict.items():
    counts, _ = np.histogram(data, bins=bins)
    distribution_counts[layer] = counts

# Plotting
plt.figure(figsize=(10, 6))

# Different colors for each layer
colors = ['blue', 'orange', 'green', 'red']

for i, (layer, counts) in enumerate(distribution_counts.items()):
    if i % 2 == 0:
        label = f"{i//2} K cache"
    else:
        label = f"{i//2} V cache"
    
    plt.plot(bins[:-1], counts, marker='o', label=label, color=colors[i])

plt.title('Value Distribution by Layer')
plt.xlabel('Value Range')
plt.ylabel('Count')
plt.xticks(bins)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and display the plot
plt.savefig("value_distribution.png")
plt.show()  # Display the plot
plt.close()  # Close the figure to free resources