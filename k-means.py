import numpy as np
import matplotlib.pyplot as plt
import pymysql
import tools
from pyecharts.charts import Bar, Line, Tab, Timeline
import pyecharts
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
import random

def fetch_data(tyear):
    conn = pymysql.connect(host='localhost', user="root", passwd="***", db="valuechain")
    cur = conn.cursor()
    cur.execute(
        'select distinct comp_info.Symbol, firmpos_info.province, firmpos_info.city, firmpos_info.district, firmpos_info.Lng, firmpos_info.Lat, supplier_data.Conumb, partner_information.province, partner_information.city, partner_information.district, partner_information.GDLng, partner_information.GDLat, supplier_data.Suplpa  from comp_info, supplier_data, partner_information, firmpos_info where comp_info.Symbol = supplier_data.Scode and comp_info.Symbol=firmpos_info.FirmID  and year(comp_info.EndDate) = supplier_data.tYear and supplier_data.tYear=firmpos_info.tYear and supplier_data.Conumb<>"" and partner_information.Conumb = supplier_data.Conumb and supplier_data.tYear='+str(tyear)+';')
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
            value = res[12]
            if ch_province != '' and fa_province != '':
                link_list.append([ch, ch_province, ch_city, ch_district, ch_wgspos, fa, fa_province, fa_city, fa_district, fa_wgspos, value])
        cur.close()
        conn.close()
        geo_cities_coords = {}
        pos = []
        wz = {}
        link = []
        values = []
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
            value = i[10]
            vs = 1
            if ch not in geo_cities_coords.keys():
                if ch_wgspos[0] is not None and ch_province in csj:
                    geo_cities_coords[ch] = ch_wgspos
                    values.append([ch, 0])
                    pos.append(ch_wgspos)
                    wz[ch] = len(values) - 1
                else:
                    vs = 0
            if fa not in geo_cities_coords.keys():
                if fa_wgspos[0] is not None and fa_province in csj:
                    geo_cities_coords[fa] = fa_wgspos
                    values.append([fa, 0])
                    pos.append(fa_wgspos)
                    wz[fa] = len(values) - 1
                else:
                    vs = 0

            if vs:
                link.append([fa, ch])
        return np.array(pos), wz, geo_cities_coords, values, link
    return 0


def t_kmeans(tyear):
    plt.rcParams['figure.figsize'] = (16, 9)

    # Creating a sample dataset with 4 clusters
    X, wz, geo_cities_coords, values, link = fetch_data(tyear)
    user_function = lambda point1, point2: tools.geo_distance(point1, point2)
    metric = distance_metric(type_metric.USER_DEFINED, func=user_function)

    # create K-Means algorithm with specific distance metric
    start_centers = []
    avg = []
    for j in range(0, 2):
        avg.append(sum(list(X[:, j])) / len(values))
    for i in range(0, 8):
        ts = []
        for j in range(0, len(avg)):
            aa = float(avg[j]) + float(random.randint(-100, 100) / 100) * 2
            if aa < 0:
                ts.append(0)
            else:
                ts.append(aa)
        start_centers.append(ts)
    sample = X

    kmeans_instance = kmeans(sample, start_centers, metric=metric)
    color = np.ones(len(wz))
    # run cluster analysis and obtain results
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    for i in range(0, len(clusters)):
        for j in range(0, len(clusters[i])):
            color[clusters[i][j]] = i

    return clusters, kmeans_instance.get_centers(), geo_cities_coords, values, link


if __name__ == "__main__":
    InitOpts = pyecharts.options.InitOpts(
        width="1900px",
        height="1000px",
    )
    timeline = Timeline(init_opts=InitOpts)
    for tyear in range(2010, 2021):
        clusters, centers, geo_cities_coords, values, link = t_kmeans(tyear)
        valuess = []
        for i in range(0, len(clusters)):
            valuess.append([])
            for j in range(0, len(clusters[i])):
                valuess[i].append(values[clusters[i][j]])
        centers_valuess = []
        for i in range(0, len(centers)):
            geo_cities_coords['Center' + str(i)] = centers[i]
            centers_valuess.append(['Center' + str(i), 100])

        timeline.add(
            tools.geo_scatter(geo_cities_coords, valuess, link, 'pos_kmeans/pos_kmeans_csj_' + str(tyear) + '.html', 'china-cities'
                              ), str(tyear))
    timeline.render('pos_kmeans_csj.html')
