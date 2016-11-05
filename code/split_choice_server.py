import simplejson as son
from bottle import Bottle, route, run, request, response, get, post

app = Bottle()

@app.route("/index.html")
def index():
    return '<a href="/hello">Hello world</a>'

if __name__ == "__main__":
    run(app, host='localhost', port=8080)
