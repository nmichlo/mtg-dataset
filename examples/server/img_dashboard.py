#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import base64
import io
from uuid import uuid4
from collections import defaultdict
import imageio

import flask
from flask import request
import requests

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

from server.util.collections import CircularQueue


# ========================================================================= #
# OPTIONS                                                                   #
# ========================================================================= #


PORT = 7777
IMAGE_FORMAT = 'jpg'
REFRESH_MS = 1001
ROW_HEIGHT = 204
ROW_SIZE = 9

# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def send_images(images, address='localhost'):
    try:
        return requests.post(f'http://{address}:{PORT}/api/image', files={
            key: imageio.imwrite('<bytes>', image, format='jpg')
            for key, image in images.items()
        })
    except:
        pass

def clear_all_images(address='localhost'):
    try:
        return requests.delete(f'http://{address}:{PORT}/api/images')
    except:
        pass

# ========================================================================= #
# SERVER                                                                    #
# ========================================================================= #


if __name__ == '__main__':

    # app & dashboard
    server = flask.Flask(__name__)
    app = dash.Dash(__name__, server=server, routes_pathname_prefix='/')

    # data storage
    IMAGE_QUEUE = defaultdict(lambda: CircularQueue(ROW_SIZE))
    UUID_TO_IMAGE = {}

    # helper function to store an image in the database, and generate and html component
    def push_list_image(key, img, serve=True):
        uuid = str(uuid4())
        img = imageio.imwrite('<bytes>', img, IMAGE_FORMAT)

        if serve:
            UUID_TO_IMAGE[uuid] = img
            src = f'/api/image/{uuid}'
        else:
            img = base64.b64encode(img).decode()
            src = f'data:image/{IMAGE_FORMAT};base64,{img}'

        replaced = IMAGE_QUEUE[key].append(dict(
            uuid=uuid,
            elem=html.Img(
                src=src,
                style={'width': 'auto', 'height': f'{ROW_HEIGHT}px'}
            )
        ))

        if replaced:
            if replaced['uuid'] in UUID_TO_IMAGE:
                del UUID_TO_IMAGE[replaced['uuid']]

    # upload images to the database
    @server.route('/api/image', methods=['POST'])
    def upload_image():
        if request.method == 'POST':
            if not request.files:
                return '[ERROR] no media in request'
            for key, file in request.files.items():
                try:
                    img = imageio.imread(file.read())
                    push_list_image(key, img)
                except:
                    pass
            return '[SUCCESS]'

    # retrieve a specific image in the database
    @server.route('/api/image/<uuid>', methods=['GET'])
    def get_image(uuid):
        try:
            if uuid in UUID_TO_IMAGE:
                fp = io.BytesIO(UUID_TO_IMAGE[uuid])
                return flask.send_file(fp, mimetype='image/jpeg') #, cache_timeout=REFRESH_MS/1000*ROW_SIZE*2)
            return 'MISSING'
        except Exception as e:
            return 'ERROR'

    # retrieve a specific image in the database
    @server.route('/api/images', methods=['DELETE'])
    def clear_images():
        IMAGE_QUEUE.clear()
        UUID_TO_IMAGE.clear()
        return 'SUCCESS'

    # refresh with new content
    @app.callback(Output('image-list', 'children'), [Input('interval-component', 'n_intervals')])
    def update_metrics(n):
        rows = []

        for name, image_elems in IMAGE_QUEUE.items():
            # append row
            rows.append(html.Plaintext(name))
            rows.append(html.Div([img['elem'] for img in image_elems]))

        # display
        return rows if rows else html.Plaintext('No images uploaded!')

    # layout of the home page
    app.layout = html.Div([
        html.Div(id='image-list'),
        dcc.Interval(id='interval-component', interval=REFRESH_MS)
    ])

    # start the server!!!
    server.run(debug=False, host='0.0.0.0', port=PORT)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
