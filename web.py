import json
from urllib.parse import parse_qs
from wsgiref import simple_server
from predict import predict


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
    text = parse_qs(query_string).get('text', ('', ))[0]
    prediction = predict(text, 50)
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
