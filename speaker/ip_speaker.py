import json
import base64
import requests

def text_to_tts_payload(text: str) -> dict:
    # 1. 文本转 GB2312 bytes
    gb_bytes = text.encode('gb2312', errors='ignore')
    # 2. GB2312 bytes 直接 Base64（设备示例也是这么做的）
    b64_str = base64.b64encode(gb_bytes).decode('ascii')
    return {"tts": {"data": b64_str}}

def send_tts(text: str, host: str = "192.168.3.29", port: int = 80):
    """
    发送文本到IP音箱进行语音播报
    
    Args:
        text: 要播报的文本内容
        host: IP音箱的IP地址，默认为192.168.3.29
        port: IP音箱的端口，默认为80
        
    Returns:
        dict: 音箱返回的响应数据
    """
    # 生成 JSON 字符串
    payload_dict = text_to_tts_payload(text)
    payload_str = json.dumps(payload_dict, ensure_ascii=False)
    # 将 JSON 字符串按 GB2312 编码
    payload_bytes = payload_str.encode('gb2312')
    url = f"http://{host}:{port}/v3"
    headers = {
        # Content-Type 可以不带 charset，默认为 GB2312 解析
        "Content-Type": "application/json"
    }
    
    print(f"发送文本到IP音箱({host}:{port}): {text[:30]}...")
    resp = requests.post(url, headers=headers, data=payload_bytes)
    resp.raise_for_status()
    return resp.json()

if __name__ == "__main__":
    # 测试用例
    sample_text = "这是一个姿势检测系统的测试消息"
    # 只发送并打印返回
    print(f"测试发送文本: {sample_text}")
    result = send_tts(sample_text)
    print("设备返回：", result)
