import json
import os
from urllib.parse import parse_qs
from wsgiref import simple_server
from predict import predict

APIKEY = os.environ.get('APIKEY', '')
CONTENT_403 = b'{"message": "You are not allowed to use this API."}'


def application(environ, start_response):
    if environ['PATH_INFO'] == '/':
        status, headers, content = index()
    elif environ['PATH_INFO'] == '/correct':
        status, headers, content = correct(environ.get('QUERY_STRING', ''))
    else:
        status, headers, content = not_found()

    headers.append(('Content-Lenght', str(len(content))))
    start_response(status, headers)
    return [content]


def correct(query_string):
    qs = parse_qs(query_string)
    apikey = qs.get('apikey', ('', ))[0]
    if apikey != APIKEY:
        return '403 Forbidden', [('Content-Type', 'application/json')], CONTENT_403

    text = qs.get('text', ('', ))[0]
    prediction = predict(text)
    content = bytes(json.dumps(prediction, ensure_ascii=False), encoding='UTF=8')
    return '200 OK', [('Content-Type', 'application/json')], content


def index():
    with open('index.html', mode='rb') as fp:
        content = fp.read()
    return '200 OK', [('Content-Type', 'text/html')], content


def not_found():
    return '404 Not Found', [('Content-Type', 'text/plain')], b'requested resource is not found'


if __name__ == '__main__':
    with simple_server.make_server('', 9310, application) as httpd:
        httpd.serve_forever()
