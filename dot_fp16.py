import numpy as np
import struct

def fp32_to_fp16(x):
    return np.float16(x)

def fp16_to_fp32(x):
    return np.float32(x)

def hex2fp16(hex_str):
    """将16位十六进制字符串转换为float16"""
    # 移除'0x'前缀（如果有的话）
    hex_str = hex_str.replace('0x', '')
    # 将16位十六进制转换为16位整数
    int_val = int(hex_str, 16)
    # 使用numpy的view方法将整数解释为float16
    return np.array([int_val], dtype=np.uint16).view(np.float16)[0]

def dot_product_fp16(vec1, vec2, acc_bit, rounding_mode='rne', detail=True):
    # 直接使用fp16进行乘法运算
    products_fp32 = np.float32(vec1) * np.float32(vec2)
    
    # 分解为指数和尾数
    mantissas, exponents = np.frexp(products_fp32)
    
    # 找到最大指数
    max_exponent = np.max(exponents)
    
    # 打印最大指数和舍入模式（10进制整数）
    rounding_mode_full = {
        'rne': 'round-to-nearest-even',
        'rz': 'round-towards-zero',
        'ru': 'round-up',
        'rd': 'round-down'
    }.get(rounding_mode, 'unknown')
    
    if detail:
        print(f"最大指数: {max_exponent}, 舍入模式: {rounding_mode_full}")
    
    # 对齐到最大指数
    aligned_mantissas = mantissas * np.power(2.0, exponents - max_exponent)
    
    if detail:
        print("对齐后的尾数（二补码），| 为累加器截断: ")
        for i, (m, e) in enumerate(zip(aligned_mantissas, exponents)):
            shift = max_exponent - e
            sign = '-' if m < 0 else '+'
            binary = np.binary_repr(int(m * (2**30)), width=31)  # 使用31位来表示尾数
            
            if shift > 0:  # 右移
                shifted_out = binary[-shift:]
                binary = '_' * shift + binary[:-shift]
            elif shift < 0:  # 左移
                shifted_out = ''
                binary = binary[-shift:].ljust(31, '0')
            else:  # 不移位
                shifted_out = ''
            
            # 确保二进制表示至少有acc_bit位
            if len(binary) < acc_bit:
                binary = binary.zfill(acc_bit)
            
            # 在acc_bit位置添加标记
            marked_binary = binary[:acc_bit] + ' | ' + binary[acc_bit:]
            
            print(f"  mantissa {i} (sign={sign}, exp={e:2d}, 右移{shift:2d}位): {marked_binary}{shifted_out}{' (LSB)' if shifted_out else ''}")
    
    # 根据acc_bit移位和舍入
    scale = 2 ** (acc_bit - 1)
    if rounding_mode == 'rne':
        rounded_mantissas = np.round(aligned_mantissas * scale) / scale
    elif rounding_mode == 'rz':
        rounded_mantissas = np.floor(aligned_mantissas * scale) / scale
    elif rounding_mode == 'ru':
        rounded_mantissas = np.ceil(aligned_mantissas * scale) / scale
    elif rounding_mode == 'rd':
        rounded_mantissas = np.floor(aligned_mantissas * scale) / scale
    else:
        raise ValueError("Invalid rounding mode. Choose from 'rne', 'rz', 'ru', 'rd'.")
    
    # 累加
    sum_mantissa = np.sum(rounded_mantissas)
    
    # 重构最终结果
    result = np.ldexp(sum_mantissa, max_exponent)
    
    if not detail:
        numpy_dot = np.dot(vec1.astype(np.float32), vec2.astype(np.float32))
        difference = result - numpy_dot
        print(f"{acc_bit},{rounding_mode},{result},{numpy_dot},{difference}")
    
    return result


def float_to_hex(f):
    return f'0x{struct.unpack("<I", struct.pack("<f", f))[0]:08X}'


def test_dot_to_numpy():
    # 使用hex2fp16函数初始化vec1和vec2
    vec1 = np.array([hex2fp16(x) for x in ['0x303C', '0x3B36', '0xBA18', '0xB1D3', '0x3B70', '0x394D', '0x3447', '0x310F']])
    vec2 = np.array([hex2fp16(x) for x in ['0x38C0', '0x2E65', '0x2EDE', '0x3035', '0x3B00', '0xBB11', '0xB501', '0xBBC3']])
    # 使用numpy.dot计算fp32结果
    result_numpy = np.dot(vec1.astype(np.float32), vec2.astype(np.float32))
    print("Initial vectors:")
    print("vec1:", vec1)
    print("vec2:", vec2)
    
    for acc_bit in range(22, 32):
        print(f"\nacc_bit: {acc_bit}")
        # 使用dot_product_fp16计算结果
        result_fp16 = dot_product_fp16(vec1, vec2, acc_bit)

        # 计算差异
        difference = abs(result_fp16 - result_numpy)
        
        # 打印结果（十进制和十六进制）
        print(f"dot_product_fp16: {result_fp16:.8f} ({float_to_hex(result_fp16)})")
        print(f"numpy_dot:        {result_numpy:.8f} ({float_to_hex(result_numpy)})")
        print(f"difference:       {difference:.8f} ({float_to_hex(difference)})")

if __name__ == "__main__":
    print("\nComparing dot_product_fp16 with numpy.dot:")
    test_dot_to_numpy()