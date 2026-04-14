import json
import threading
import time
import os
from flask import Flask, request
from flask_redis import FlaskRedis
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
import queue

from utils import predict_utils as pu

app = Flask(__name__)
# MySQL所在主机名
HOSTNAME = "10.128.2.23"
# MySQL监听的端口号
PORT = 3306
# 连接MySQL的用户名
USERNAME = "root"
# 连接MySQL的密码
PASSWORD = "114514Homo?"
# MySQL上创建的数据库名称
DATABASE = "drug"
# 通过修改以下代码来操作不同的SQL比写原生SQL简单很多 --》通过ORM可以实现从底层更改使用的SQL
app.config[
    'SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4"
# 连接数据库
db = SQLAlchemy(app)
# 预测队列
queue = queue.Queue(maxsize=100)
# 配置 Redis 连接 URL
app.config['REDIS_URL'] = "redis://:gdpu20240327@10.128.2.23:6379/0"

# 初始化 Flask-Redis 扩展
redis_client = FlaskRedis(app)

with app.app_context():
    with db.engine.connect() as conn:
        rs = conn.execute(text("select 1"))
        print(rs.fetchall())  # 输出 [(1,)] 说明连接成功


@app.route('/aaa', methods=['GET'])
def aaa():
    print("1")
    current_path = os.getcwd()
    print(f"当前工作目录: {current_path}")
    return "1"
   

@app.route('/predict/<drug_smiles>', methods=['GET'])
def predict(drug_smiles):
    queue.put(drug_smiles)
    with app.app_context():
        with db.engine.connect() as conn:
            sql = text("SELECT GeneSymbol,ChemicalID FROM CTD_chem_gene_ixns WHERE SMILES = :smiles GROUP BY GeneSymbol,ChemicalID")
            print(sql,f"drug_smiles: {drug_smiles}")
            ctd = conn.execute(sql, {"smiles": drug_smiles})
    ctdListQuery = ctd.fetchall()
    ctdList = []
    if ctdListQuery.__len__() > 0:
        for ctd in ctdListQuery:
            ctdList.append(ctd.GeneSymbol)#获取待预测smiles有关和基因和CHemicalID
            drug_id = ctd.ChemicalID
            print(drug_id)
        output = pu.predict(drug_smiles, ctdList, drug_id)
        outputjson = output.to_json(orient='records')
        outputlist = json.loads(outputjson)
        
        # 将分数转换为浮点数以便正确排序
        for item in outputlist:
            item['Sample_association_score'] = float(item['Sample_association_score'])
            
        # 按分数从高到低排序
        outputlist = sorted(outputlist, key=lambda x: x['Sample_association_score'], reverse=True)
        
        # 去除side_effect_id重复的项，保留分数最高的
        unique_side_effects = {}
        for item in outputlist:
            side_effect = item['side_effect_id']
            if side_effect not in unique_side_effects:
                unique_side_effects[side_effect] = item
                
        # 转换回列表并保持排序
        outputlist = list(unique_side_effects.values())
            
        output=''
        for i in outputlist:
            output += i['drug_id'] + '\t' + i['side_effect_id'] +'\t' + str(i['Sample_association_score']) +'\t'
        output = output.strip('\t')
    else:
        output = "暂不支持该药物的预测"
    #预测结果存储到redis
    redis_client.set(drug_smiles, output)
    return output
    # data = {"data": output}
    # return data


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
    # app.run(host='10.128.2.23', port=8081)
