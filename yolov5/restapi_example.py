#!/usr/bin/env python
# encoding: utf-8
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
import os


app = Flask(__name__)
@app.route('/')
def index():
    
    return jsonify({'name': 'alice',
                    'email': 'alice@outlook.com'})


# @app.route('/')
# def index():
#     return render_template('search.html')


# @app.route('/detect/')
# def logs():
#     filenames = os.listdir('detect')
#     return render_template('logs.html', files=filenames)

# @app.route('/detect/<path:foldername>')
# def logs(foldername):
#     filenames = os.listdir('detect/' + foldername)
#     return render_template('logs.html', files=filenames)

# @app.route('/detect/<path:filename>')
# def log(filename):
#     filenames = os.listdir('detect/' + filename )
#     return send_from_directory(
#         os.path.abspath('detect/' + filenames ),
#         filename,
#         as_attachment=True
#     )



# @app.route('/<string:name>/')
# def hello(name):
#     print(name)
#     path = os.getcwd()
#     print(path)
#     print(type(path))

#     os.chdir('detect')

#     print(os.getcwd())

#     return {"Hello ": name}


# # @app.route('/<string:name>/', methods=['GET'])
# # def query_records(name):
# #     name = request.args.get('name')
# #     print (name)
# #     # with open('/tmp/data.txt', 'r') as f:
# #     #     data = f.read()
# #     #     records = json.loads(data)
# #     #     for record in records:
# #     #         if record['name'] == name:
# #     #             return jsonify(record)
# #         # return jsonify({'error': 'data not found'})
# #     return jsonify({'Hello': name})

# @app.route('/', methods=['PUT'])
# def create_record():
#     record = json.loads(request.data)
#     with open('/tmp/data.txt', 'r') as f:
#         data = f.read()
#     if not data:
#         records = [record]
#     else:
#         records = json.loads(data)
#         records.append(record)
#     with open('/tmp/data.txt', 'w') as f:
#         f.write(json.dumps(records, indent=2))
#     return jsonify(record)

# @app.route('/', methods=['POST'])
# def update_record():
#     record = json.loads(request.data)
#     new_records = []
#     with open('/tmp/data.txt', 'r') as f:
#         data = f.read()
#         records = json.loads(data)
#     for r in records:
#         if r['name'] == record['name']:
#             r['email'] = record['email']
#         new_records.append(r)
#     with open('/tmp/data.txt', 'w') as f:
#         f.write(json.dumps(new_records, indent=2))
#     return jsonify(record)
    
# @app.route('/', methods=['DELETE'])
# def delte_record():
#     record = json.loads(request.data)
#     new_records = []
#     with open('/tmp/data.txt', 'r') as f:
#         data = f.read()
#         records = json.loads(data)
#         for r in records:
#             if r['name'] == record['name']:
#                 continue
#             new_records.append(r)
#     with open('/tmp/data.txt', 'w') as f:
#         f.write(json.dumps(new_records, indent=2))
#     return jsonify(record)

app.run(debug=True)