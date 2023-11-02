from _datetime import datetime
import json
from logging import getLogger, config
import logging

from websocket_server.websocket_server import WebsocketServer

from boatrace.server.service.AbstractRequestDispatcher import AbstractRequestDispatcher
from boatrace.util.PropertyUtil import PropertyUtil
import sys
from boatrace.server.service.JsonRequestDispatcher import JsonRequestDispatcher


class BoatWebsocketServer():
    def __init__(self, host, port, dispatcher:AbstractRequestDispatcher):
        self._logger_ = getLogger('server')
        self.server = WebsocketServer(host, port, loglevel=logging.DEBUG)
        self._dispatcher_:AbstractRequestDispatcher = dispatcher

    # クライアント接続時に呼ばれる関数
    def new_client(self, client, server):
        self._logger_.info("new client connected and was given id {}".format(client['id']))
        # 全クライアントにメッセージを送信
        # self.server.send_message_to_all("hey all, a new client has joined us")
        prop:PropertyUtil = PropertyUtil.getInstance()
        prop.reload()

    # クライアント切断時に呼ばれる関数
    def client_left(self, client, server):
        self._logger_.info("client({}) disconnected".format(client['id']))

    # クライアントからメッセージを受信したときに呼ばれる関数
    def message_received(self, client, server, message):
        # self._logger_.debug("client({}) said: {}".format(client['id'], message))
        
        response:str = self._dispatcher_.dispatch(message)
        
        # send response
        self.server.send_message(client, response)
    
    # サーバーを起動する
    def run(self):
        self._logger_.info('server started.')
        # クライアント接続時のコールバック関数にself.new_client関数をセット
        self.server.set_fn_new_client(self.new_client)
        # クライアント切断時のコールバック関数にself.client_left関数をセット
        self.server.set_fn_client_left(self.client_left)
    # メッセージ受信時のコールバック関数にself.message_received関数をセット
        self.server.set_fn_message_received(self.message_received) 
        self.server.run_forever()
        
def logSetup():
    """
    loggin環境を定義する
    """ 
    prop = PropertyUtil.getInstance()
    with open(prop.getProperty('file_python_log_config'), 'r', encoding='utf-8') as f:
        log_config = json.load(f)
        
    # ファイル名をタイムスタンプで作成
    log_file_name = prop.getProperty('file_python_log')
    log_config["handlers"]["fileHandler"]["filename"] = log_file_name.format(datetime.utcnow().strftime("%Y%m%d"))

    config.dictConfig(log_config)

def main(argv):
    #propfile_expr10:str = 'C:/Dev/workspace/Oxygen/pod_boatrace/properties/expr10/expr10.properties'
    #propfile_model:str = 'C:/Dev/workspace/Oxygen/pod_boatrace/properties/expr10/model.properties'

    propfile_expr10:str = argv[1]
    propfile_model:str = argv[2]
        
    
    prop:PropertyUtil = PropertyUtil.getInstance()
    
    # properties(java, python共有)
    prop.addFile(propfile_expr10)
    
    # モデル情報 (MLModelGenerator.javaが実行されるたびに生成・更新される）
    prop.addFile(propfile_model)
    
    # loggin set up
    logSetup()

    # server start
    url:str = prop.getProperty('websocket_url_python')
    token:list[str] = url.split('//')[1].split(':')
    server:BoatWebsocketServer = BoatWebsocketServer(token[0], int(token[1]), JsonRequestDispatcher())
    server.run()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
