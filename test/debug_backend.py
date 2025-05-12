import sys
import time
import threading
import argparse
from flask import request, jsonify

# 导入主应用
import main

def setup_args():
    parser = argparse.ArgumentParser(description='Debug backend functionality')
    parser.add_argument('--rtsp_url', type=str, default="rtsp://192.168.3.242:8554/live",
                       help='RTSP URL to connect to')
    parser.add_argument('--mode', type=str, choices=['rtsp', 'tts_test'], default='rtsp',
                       help='Debug mode: rtsp (stream processing) or tts_test (test TTS)')
    parser.add_argument('--frames', type=int, default=0,
                       help='Number of frames to process (0 = unlimited)')
    parser.add_argument('--report_interval', type=int, default=10,
                       help='Generate posture report every N frames')
    
    return parser.parse_args()

def debug_rtsp_mode(args):
    """测试RTSP流模式"""
    print(f"===== RTSP调试模式 =====")
    print(f"RTSP URL: {args.rtsp_url}")
    print(f"帧数限制: {'无限制' if args.frames == 0 else args.frames}")
    print(f"报告间隔: 每 {args.report_interval} 帧")
    
    # 设置全局变量
    main.rtsp_url = args.rtsp_url
    
    # 启动RTSP线程
    main.rtsp_thread_running = True
    main.rtsp_thread = threading.Thread(target=main.rtsp_detection_thread)
    main.rtsp_thread.daemon = True
    main.rtsp_thread.start()
    
    print("RTSP线程已启动，等待数据...")
    time.sleep(3)  # 给RTSP线程一些时间来初始化

    # 处理指定数量的帧
    frame_count = 0
    try:
        while main.rtsp_thread_running:
            if not main.rtsp_result_queue.empty():
                result = main.rtsp_result_queue.queue[0]  # 获取但不移除
                posture_status = result.get('posture_status')
                angles = result.get('angles', {})
                
                # 提取和打印关键数据
                print(f"\nFrame #{frame_count+1}:")
                print(f"  姿势状态: {posture_status}")
                if angles:
                    print(f"  角度: 颈部={angles.get('neck', '--')}°, 身体={angles.get('avg_body', '--')}°")
                else:
                    print("  未检测到角度数据")
                
                # 定期生成姿势报告
                if frame_count % args.report_interval == 0 and posture_status == "bad":
                    print("\n===== 生成姿势报告 =====")
                    report_data = main.get_posture_report(angles, posture_status)
                    print(f"  AI建议: {report_data.get('text', '')}")
                
                frame_count += 1
                
                # 如果达到指定帧数限制，停止
                if args.frames > 0 and frame_count >= args.frames:
                    print(f"\n已处理 {args.frames} 帧，退出")
                    break
            
            time.sleep(0.1)  # 短暂休眠以减少CPU使用
            
    except KeyboardInterrupt:
        print("\n用户中断，停止处理")
    finally:
        # 停止RTSP线程
        main.rtsp_thread_running = False
        if main.rtsp_thread:
            main.rtsp_thread.join(timeout=5.0)
        print("RTSP线程已停止")

def debug_tts_test():
    """测试TTS功能"""
    print("===== TTS测试模式 =====")
    
    # 模拟执行test_tts_endpoint
    mock_request = type('obj', (object,), {
        'get_json': lambda: None,
        'json': None,
        'form': {},
        'args': {},
    })
    
    # 保存原始request
    original_request = request
    # 模拟request
    request.__class__ = mock_request.__class__
    
    try:
        print("执行TTS测试...")
        response = main.test_tts_endpoint()
        print(f"TTS测试响应: {response.json}")
    except Exception as e:
        print(f"TTS测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复原始request
        request.__class__ = original_request.__class__

if __name__ == "__main__":
    args = setup_args()
    
    print("=================================")
    print("智能姿势检测系统后端调试工具")
    print("=================================")
    
    if args.mode == 'rtsp':
        debug_rtsp_mode(args)
    elif args.mode == 'tts_test':
        debug_tts_test() 