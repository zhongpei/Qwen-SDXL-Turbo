import http.server
import socketserver
import threading
import socket
import time

OUTPUT_HTML_DIR = "output_html"


def get_local_ip():
    """ 获取当前计算机在局域网中的 IP 地址 """
    try:
        # 建立一个临时的连接，以便获取本地网络接口的IP地址
        # 这里的 'www.baidu.com' 是Google的公共DNS服务器，端口为80
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("www.baidu.com", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        print("获取本地IP地址时出错：", e)
        return None


class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # 指定要提供的文件的目录
        if OUTPUT_HTML_DIR not in self.path:
            self.path = OUTPUT_HTML_DIR + self.path
        print(f"请求的文件路径为：{self.path}")
        return http.server.SimpleHTTPRequestHandler.do_GET(self)


def _start_server(server_port):
    with socketserver.TCPServer(("", server_port), MyHttpRequestHandler) as httpd:
        print(f"HTTP 文件服务器启动在端口 {server_port}")
        httpd.serve_forever()


def start_server(server_port):
    # 在单独的线程中启动服务器
    server_thread = threading.Thread(target=_start_server, args=(server_port,))
    server_thread.daemon = True  # 设置为守护线程，这样当主程序退出时，服务器线程也会退出
    server_thread.start()


if __name__ == "__main__":
    start_server(8001)
    while True:
        time.sleep(1000000)
