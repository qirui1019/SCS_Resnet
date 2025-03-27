import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SERVER_URL = "http://10.27.251.68:5000"


def fetch_transfer_rate():
    """ 使用 requests.Session() 复用 HTTP 连接，避免端口耗尽 """
    # 创建一个全局 requests.Session 并设置连接池大小
    url = f"{SERVER_URL}/transfer_rate"
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100,
                                            max_retries=Retry(total=3, backoff_factor=0.1))  # 连接池大小 100
    session.mount("http://", adapter)

    while True:
        try:
            response = session.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"当前数据传输速率: {data['transfer_rate']}")
            else:
                print(f"请求失败，状态码: {response.status_code}")
        except requests.RequestException as e:
            print(f"请求错误: {e}")

        time.sleep(1)  # 每 1 秒请求一次


if __name__ == "__main__":
    fetch_transfer_rate()
