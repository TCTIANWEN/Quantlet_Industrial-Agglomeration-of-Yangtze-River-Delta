import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pymysql
import math
import tools
from pyecharts.charts import Bar, Line, Tab, Timeline
import pyecharts
import random
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from sklearn.manifold import TSNE


def fetch_data(tyear):
    conn = pymysql.connect(host='localhost', user="root", passwd="***", db="valuechain")
    cur = conn.cursor()
    cur.execute(
        'select distinct comp_info.Symbol, firmpos_info.province, firmpos_info.city, firmpos_info.district, firmpos_info.Lng, firmpos_info.Lat, supplier_data.Conumb, partner_information.province, partner_information.city, partner_information.district, partner_information.Lng, partner_information.Lat, supplier_data.Suplpa  from comp_info, supplier_data, partner_information, firmpos_info where comp_info.Symbol = supplier_data.Scode and comp_info.Symbol=firmpos_info.FirmID  and year(comp_info.EndDate) = supplier_data.tYear and supplier_data.tYear=firmpos_info.tYear and supplier_data.Conumb<>"" and partner_information.Conumb = supplier_data.Conumb and supplier_data.tYear='+str(tyear)+';')
    link_list = []
    csj = ['上海市', '安徽省', '江苏省', '浙江省']
    if cur.rowcount > 0:
        while 1:
            res = cur.fetchone()
            if res is None:
                break
            ch = res[0]
            ch_province = res[1]
            ch_city = res[2]
            ch_district = res[3]
            ch_wgspos = [res[4], res[5]]
            fa = res[6]
            fa_province = res[7]
            fa_city = res[8]
            fa_district = res[9]
            fa_wgspos = [res[10], res[11]]
            value = float(res[12])
            if ch_province != '' and fa_province != '' and value>0:
                link_list.append([ch, ch_province, ch_city, ch_district, ch_wgspos, fa, fa_province, fa_city, fa_district, fa_wgspos, value])
    cur.execute('select distinct comp_info.Symbol, firmpos_info.province, firmpos_info.city, firmpos_info.district, firmpos_info.Lng, firmpos_info.Lat, buyer_data.Conumb, partner_information.province, partner_information.city, partner_information.district, partner_information.Lng, partner_information.Lat, buyer_data.Custinc  from comp_info, buyer_data, partner_information, firmpos_info where comp_info.Symbol = buyer_data.Scode and comp_info.Symbol=firmpos_info.FirmID  and year(comp_info.EndDate) = buyer_data.tYear and buyer_data.tYear=firmpos_info.tYear and buyer_data.Conumb<>"" and partner_information.Conumb = buyer_data.Conumb and buyer_data.tYear=' + str(
                tyear) + ';')
    if cur.rowcount > 0:
        while 1:
            res = cur.fetchone()
            if res is None:
                break
            fa = res[0]
            fa_province = res[1]
            fa_city = res[2]
            fa_district = res[3]
            fa_wgspos = [res[4], res[5]]
            ch = res[6]
            ch_province = res[7]
            ch_city = res[8]
            ch_district = res[9]
            ch_wgspos = [res[10], res[11]]
            value = float(res[12])
            if ch_province != '' and fa_province != '' and value>0:
                link_list.append(
                    [ch, ch_province, ch_city, ch_district, ch_wgspos, fa, fa_province, fa_city, fa_district,
                     fa_wgspos, value])

        cur.close()
        conn.close()
        geo_cities_coords = {}
        pos = []
        wz = {}
        link = []
        values = []
        cities = []
        dict_city = {}
        for i in link_list:
            ch = i[0]
            ch_province = i[1]
            ch_city = i[2]
            ch_district = i[3]
            ch_wgspos = i[4]
            fa = i[5]
            fa_province = i[6]
            fa_city = i[7]
            fa_district = i[8]
            fa_wgspos = i[9]
            value = float(i[10])
            if ch_city is None or fa_city is None or ch_city == '[]' or fa_city == '[]':
                continue
            if ch_city not in cities:
                cities.append(ch_city)
                dict_city[ch_city] = [len(cities) - 1, {}, {}, 0, 0]
                values.append([ch_city, 100])
            if fa_city not in cities:
                cities.append(fa_city)
                dict_city[fa_city] = [len(cities) - 1, {}, {}, 0, 0]
                values.append([fa_city, 100])
            if fa_city in dict_city[ch_city][1].keys():
                dict_city[ch_city][1][fa_city] += value
            else:
                dict_city[ch_city][1][fa_city] = value
            if ch_city in dict_city[fa_city][2].keys():
                dict_city[fa_city][2][ch_city] += value
            else:
                dict_city[fa_city][2][ch_city] = value
            dict_city[ch_city][3] += value
            dict_city[fa_city][4] += value
            if [fa_city, ch_city] not in link and [ch_city, fa_city] not in link:
                link.append([fa_city, ch_city])

        city_link = []
        for i in range(0, len(cities) +1):
            city_link.append([])
            for j in range(0, 1+len(cities)):
                city_link[i].append(0)
        for i in range(0, len(cities)):
            city_link[i][len(cities)] = dict_city[cities[i]][4]
            city_link[len(cities)][i] = dict_city[cities[i]][3]
            for j in range(0, len(cities)):
                if cities[j] in dict_city[cities[i]][2].keys():
                    city_link[i][j] = dict_city[cities[i]][2][cities[j]]

        return np.array(city_link), wz, geo_cities_coords, values, link
    return 0


def distance(point1, point2):
    if point2[3 + int(point1[0])] + point1[3 + int(point2[0])]== 0:
        return float('inf')
    else:
        return min((point1[1] + point1[2])/ (point2[3 + int(point1[0])] + point1[3 + int(point2[0])]) ,
            (point2[1] + point2[2])/ (point2[3 + int(point1[0])] + point1[3 + int(point2[0])]) )


def distance2(point1, point2, norm=False):
    x = point1
    y = point2
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if all(x == zero_list) or all(y == zero_list):
        return float(1)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 1 - cos


def t_kmeans(tyear):

    # Creating a sample dataset with 4 clusters
    X, wz, geo_cities_coords, values, link = fetch_data(tyear)

    user_function = lambda point1, point2: distance2(point1, point2)
    metric = distance_metric(type_metric.USER_DEFINED, func=user_function)

    # create K-Means algorithm with specific distance metric
    sample1 = X[0:-1, 0:-1]
    sample = 0*sample1
    bb = len(sample1)
    for i in range(0, bb):
        for j in range(0, bb):
            if (X[i][bb]+X[bb][i]) ==0 :
                print(X)
            sample[i][j] = (sample1[i][j] + sample1[j][i])/(X[i][bb]+X[bb][i])
    start_centers = []

    for i in range(0, 9):
        ts = []
        for j in range(0, bb):
            aa = float(random.randint(0, 100)/100)
            ts.append(aa)
        start_centers.append(ts)


    kmeans_instance = kmeans(sample, start_centers, metric=metric)
    color = np.ones(len(values))
    # run cluster analysis and obtain results
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    for i in range(0, len(clusters)):
        for j in range(0, len(clusters[i])):
            color[clusters[i][j]] = i

    return clusters, sample, geo_cities_coords, values, link


if __name__ == "__main__":

    InitOpts = pyecharts.options.InitOpts(
        width="1900px",
        height="1000px",
    )
    delete = ['市', '地区', '蒙古自治州', '蒙古族藏族自治州', '藏族自治州', '朝鲜族自治州', '壮族苗族自治州', '布依族苗族自治州', '哈萨克自治州', '白族自治州', '回族自治州',
              '特别自治区', '柯尔克孜自治州']
    timeline = Timeline(init_opts=InitOpts)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    for tyear in range(2010, 2021):
        clusters, sample, geo_cities_coords, values, link = t_kmeans(tyear)
        data_TSNE = TSNE(learning_rate=100).fit_transform(sample)
        type_colors = ['red', 'blue', 'black', 'darkorange', 'green', 'brown', 'orangered', 'darkgray', 'deeppink']
        colors = []
        names = []
        for i in range(0, len(values)):
            names.append(values[i][0])
            colors.append('')
        num_type = len(clusters)
        for i in range(0, num_type):
            for j in range(0, len(clusters[i])):
                values[clusters[i][j]][1] = 100/(num_type-1)*i
                colors[clusters[i][j]] = type_colors[i]
                for kk in delete:
                    values[clusters[i][j]][0] = values[clusters[i][j]][0].replace(kk, '')
        timeline.add(tools.map_print(values, 'china-cities', '11/pos_kmeans_' + str(tyear) + '0.html'), str(tyear))

        plt.scatter(data_TSNE[:, 0], data_TSNE[:, 1], c=colors, s=10)
        for i in range(0, len(values)):
            plt.text(data_TSNE[i, 0]+0.005, data_TSNE[i, 1]+0.005, names[i],  fontsize=8, c=colors[i])

        plt.title('TSNE Result of '+str(tyear))
        plt.savefig('relyingrate/'+str(tyear)+'.png')
    timeline.render('k-means-city.html')
