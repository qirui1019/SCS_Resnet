import psutil
import time


def monitor_cpu_usage(interval=1):
    """
    每隔指定时间 (秒) 打印一次当前 CPU 利用率。
    """
    while True:
        cpu_usage = psutil.cpu_percent(interval=interval)
        print(f"Current CPU usage: {cpu_usage}%")
        time.sleep(interval)


if __name__ == "__main__":
    monitor_cpu_usage(1)
