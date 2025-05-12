import argparse
from speaker.ip_speaker import send_tts

def main():
    parser = argparse.ArgumentParser(description='测试IP音箱TTS功能')
    parser.add_argument('--text', type=str, default="这是一条测试消息，测试智能姿势检测系统的语音播报功能",
                       help='要播报的文本')
    parser.add_argument('--host', type=str, default="192.168.3.29",
                       help='IP音箱的IP地址')
    parser.add_argument('--port', type=int, default=80,
                       help='IP音箱的端口')
    
    args = parser.parse_args()
    
    print(f"===== IP音箱TTS测试 =====")
    print(f"目标设备: {args.host}:{args.port}")
    print(f"播报文本: {args.text}")
    
    try:
        result = send_tts(args.text, args.host, args.port)
        print(f"发送成功! 设备返回: {result}")
    except Exception as e:
        print(f"发送失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 