"""
check_env.py
环境诊断：检查 PyTorch、CUDA/GPU 是否正常，确保可以用 GPU 加速。
运行方式：python check_env.py
"""
import torch
import torchvision


def check_gpu():
    print("=" * 50)
    print("环境诊断")
    print("=" * 50)
    print(f"PyTorch 版本:    {torch.__version__}")
    print(f"Torchvision 版本: {torchvision.__version__}")
    print(f"CUDA 可用:       {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU 型号:        {torch.cuda.get_device_name(0)}")
        x = torch.rand(5, 3).cuda()
        print(f"GPU 张量测试:    通过 (shape={list(x.shape)})")
    else:
        print("警告: CUDA 不可用，将使用 CPU（速度较慢）")
        print("建议检查：驱动版本、CUDA Toolkit、PyTorch 安装命令是否匹配")

    print("=" * 50)


if __name__ == "__main__":
    check_gpu()
