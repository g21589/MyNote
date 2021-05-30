# -*- coding: utf-8 -*-

import logging
import flask
import requests
from flask import Flask, jsonify

app = Flask(__name__)

method_requests_mapping = {
    'GET': requests.get,
    'HEAD': requests.head,
    'POST': requests.post,
    'PUT': requests.put,
    'DELETE': requests.delete,
    'PATCH': requests.patch,
    'OPTIONS': requests.options,
}

@app.route('/', methods=['GET'])
def hello_world():
    response = jsonify(message='Simple server is running')
    #response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/proxy/<path:url>', methods=method_requests_mapping.keys())
def proxy(url):
    #app.logger.info(f'[ST] Proxy {url}')
    requests_function = method_requests_mapping[flask.request.method]
    request = requests_function(url, stream=True, params=flask.request.args)
    response = flask.Response(flask.stream_with_context(request.iter_content()),
                              content_type=request.headers['content-type'],
                              status=request.status_code)
    response.headers['Access-Control-Allow-Origin'] = '*'
    #app.logger.info(f'[ED] Proxy {url}')
    return response

if __name__ == '__main__':
    
    #handler = logging.FileHandler('server.log')
    #app.logger.addHandler(handler)
    
    app.run(host='localhost', port=8080, threaded=True, debug=True)
