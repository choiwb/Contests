import requests
import pandas as pd
import numpy as np
import os
import time
import datetime
import json

start_time = time.time()

csv_list = [file for file in os.listdir() if file.endswith("boxoffice.csv")]

for i, data in enumerate(csv_list):
    if i == 0:
        df = pd.read_csv(data)
    else:
        small_df = pd.read_csv(data)
        df = pd.concat([df, small_df]).reset_index(drop=True)

df["관객수"] = df["관객수"].str.replace(",","")
df["관객수"] = df["관객수"].astype(int)
movie_df = df[df["관객수"] > 10000].reset_index(drop=True)
movie_df = movie_df.drop(movie_df.index[[3191]])
movie_df = movie_df.drop_duplicates("영화명").reset_index(drop=True)
movie_df.to_csv("movie.csv",encoding="utf-8")

movieNm = movie_df["영화명"][:100]

def get_movie_data(movieNm):
    url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieList.json"
    params = {"key":"92666a5a1a0aac363a32550004d99e5c", "movieNm":movieNm}
    r = requests.get(url, params=params)
    return r.json()

def make_movie_df(movieNm):
    movie_df = pd.DataFrame(columns = ["movieCd", "movieNm", "director","prdtYear", "openDt", "typeNm", "repNationNm", "repGenreNm", "companyNm"])
    for i in list(movieNm):
        try:
            for data in get_movie_data(i)['movieListResult']['movieList']:
                if len(data["directors"]) >= 2:
                    director = data["directors"][0]["peopleNm"]
                elif len(data["directors"]) == 1:
                    director = data["directors"][0]["peopleNm"]
                if len(data["companys"]) >= 2:
                    companyNm = data["companys"][0]["companyNm"]
                elif len(data["companys"]) == 1:
                    companyNm = data["companys"][0]["companyNm"]
                movie_df.loc[len(movie_df)] = [
                    data["movieCd"],
                    data["movieNm"],
                    director,
                    data["prdtYear"],
                    data["openDt"],
                    data["typeNm"],
                    data["repNationNm"],
                    data["repGenreNm"],
                    companyNm
                ]
        except:
            print(i)
    return movie_df

movie_info_df = make_movie_df(movieNm)

movie_info_df = movie_info_df[~movie_info_df["movieNm"].str.contains("시네마정동")].reset_index(drop=True)
movie_info_df = movie_info_df[movie_info_df["repNationNm"] != "기타"].reset_index(drop=True)
movie_info_df = movie_info_df.drop_duplicates("movieCd").reset_index(drop=True)
movie_info_df = movie_info_df.drop_duplicates("movieNm").reset_index(drop=True)
movie_info_df = movie_info_df[movie_info_df["openDt"] != ""].reset_index(drop=True)
movie_info_df.to_csv("movie_info.csv",encoding="utf-8")

print("movie_info_df의 행렬 형태:",movie_info_df.shape)

#movieCd = movie_info_df["movieCd"][:3000]
movieCd = movie_info_df["movieCd"]
#movieCd1 = movie_info_df["movieCd"][3000::]

def get_movie_detail(movieCd):
    url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieInfo.json"
    params = {"key":"f267fc95b611534c5df04496dd82e2ac", "movieCd":movieCd}
    r = requests.get(url, params=params)
    return r.json()

def make_movie_detail_df(movie_info_df):
    movie_detail_df = pd.DataFrame(columns=["movieCd",
                                            "movieNm",
                                            "showTm",
                                            "watchGradeNm",
                                            "actor_1",
                                            "actor_2",
                                            "actor_3",
                                            "companyNm"
                                            ])
    #for i in list(movieCd1):
    for i in list(movieCd):
        try:
            data = get_movie_detail(i)['movieInfoResult']['movieInfo']
            actor_list = []
            if len(data["audits"]) >= 2:
                watchGradeNm = data["audits"][0]["watchGradeNm"]
            elif len(data["audits"]) == 1:
                watchGradeNm = data["audits"][0]["watchGradeNm"]
            if len(data["companys"]) >= 2:
                companyNm = data["companys"][0]["companyNm"]
            elif len(data["companys"]) == 1:
                companyNm = data["companys"][0]["companyNm"] 
            if len(data["actors"]) >= 3:
                actor_list = [
                    data["actors"][0]["peopleNm"],
                    data["actors"][1]["peopleNm"],
                    data["actors"][2]["peopleNm"]
                ]
            else:
                for i in range(len(data["actors"])):
                    actor_list.append(data["actors"][i]["peopleNm"])
                for i in range(3-len(data["actors"])):
                    actor_list.append("")
            movie_detail_df.loc[len(movie_detail_df)] = [
                data["movieCd"],
                data["movieNm"],
                data["showTm"],
                watchGradeNm,
                *actor_list,
                companyNm
            ]
        except:
             print(i)
    return movie_detail_df

movie_detail_df = make_movie_detail_df(movieCd)

print('movie_detail_df의 행렬 형태:', movie_detail_df.shape)
#print(movie_detail_df)
#movie_detail1_df = make_movie_detail_df(movieCd1)
#movie_detail = pd.concat([movie_detail_df, movie_detail1_df]).reset_index(drop=True)

#boxoffice_df = movie_info_df.merge(movie_detail, left_on="movieCd", right_on="movieCd")[[
boxoffice_df = movie_info_df.merge(movie_detail_df, left_on="movieCd", right_on="movieCd")[[
        "movieCd",
        "movieNm_x",
        "director",
        "openDt",
        "prdtYear",
        "repNationNm",
        "repGenreNm",
        "showTm",
        "watchGradeNm",
        "actor_1",
        "actor_2",
        "actor_3",
        "companyNm_y"
    ]]
boxoffice_df = boxoffice_df.rename(columns={"movieNm_x":"movieNm", "companyNm_y":"companyNm"})
print("최종 모양은:",boxoffice_df.shape)

boxoffice_df.to_csv("boxoffice.csv",encoding="utf-8")

import requests
import pandas as pd
import numpy as np
import time
import json
import datetime
from bs4 import BeautifulSoup

def send_slack(channel, username, icon_emoji, message):
    base_url = "https://hooks.slack.com/services/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"
    payload = {
        "channel": channel,
        "username": username,
        "icon_emoji": icon_emoji,
        "text": message
    }
    response = requests.post(base_url, data=json.dumps(payload))
    print(response.content)

def slack(function):
    def wrapper(*args, **kwargs):
        name = function.__name__
        start_time = time.time()
        current_time = str(datetime.datetime.now())
        send_slack("movie", "databot", ":ghost:", "작업 실행 - {time}".format(time=current_time))

        try:
            result = function(*args, **kwargs)
            current_time = str(datetime.datetime.now())
            end_time = time.time()
            send_slack("movie", "databot", ":ghost:",
                       "작업 끝 - 걸린시간{time(s)}: {time}s.".format(time=int(end_time - start_time)))

        except:
            send_slack("movie", "databot", ":ghost:", "오류 발생.")
        return result

    return wrapper

boxoffice_df = pd.read_csv("data/boxoffice.csv",index_col=0)

@slack
def get_audience(boxoffice_df):
    final_audience_df = pd.DataFrame(columns=["movieCd","preview_audience",
                                        "d1_audience","d2_audience","d3_audience","d4_audience","d5_audience","d6_audience","d7_audience",
                                        "d1_screen","d2_screen","d3_screen","d4_screen","d5_screen","d6_screen","d7_screen",
                                        "d1_show","d2_show","d3_show","d4_show","d5_show","d6_show","d7_show",
                                        "d1_seat","d2_seat","d3_seat","d4_seat","d5_seat","d6_seat","d7_seat",
                                        "audience"])
    for j in list(boxoffice_df["movieCd"]):
        try:
            data = {"code":j, "sType": "stat"}
            headers = {"Accept-Encoding":"gzip, deflate",
                    "Accept-Language":"ko-KR,ko;q=0.8,en-US;q=0.6,en;q=0.4",
                    "Connection":"keep-alive",
                    "Content-Length":24,
                    "Content-Type":"application/x-www-form-urlencoded",
                    "Cookie":"ACEFCID=UID-57832AC44039B8B57BE3DF6B; JSESSIONID=S2J8XM7QzvvL8t56GVYqdfpVH6cd1X28XC39wnTQGy7yLGnWhQFn!1412368483!-1881944657",
                    "Host":"www.kobis.or.kr",
                    "Origin":"http://www.kobis.or.kr",
                    "Referer":"http://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieList.do",
                    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36",
                    "X-Requested-With":"XMLHttpRequest"}
            #r = requests.post("http://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieDtl.do",data=data,headers=headers)
            r = requests.post("http://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieDtl.do",data=data)

            dom = BeautifulSoup(r.content, "html.parser")
            dom1 = dom.select("td.right")
            dom2 = dom.select("td.bgt2")
            preview_audience = dom1[3].text
            audience_list = []
            for i in range(9,58,7):
                audience_list.append(dom1[i].text)
            screen_list = []
            for i in range(4,32,4):
                screen_list.append(dom2[i].text)
            show_list = []
            for i in range(5,33,4):
                show_list.append(dom2[i].text)
            seat_list = []
            for i in range(7,35,4):
                seat_list.append(dom2[i].text)
            audience = dom.select("td")[7].text
            final_audience_df.loc[len(final_audience_df)] = [
                j,
                preview_audience,
                *audience_list,
                *screen_list,
                *show_list,
                *seat_list,
                audience,
            ]
            print('성공')
        except:
            print('get_audience에서 실패')
            print(j)
    return final_audience_df

final_audience_df = get_audience(boxoffice_df)
final_audience_df = pd.read_csv("data/final_audience.csv", index_col=0)

final_audience_df["movieCd"] = final_audience_df["movieCd"]

final_audience_df["audience"] = final_audience_df["audience"].dropna().apply(lambda x: int(str(x).split("(")[0].replace(",", "")) if str(x).split("(")[0].replace(",", "") != " " else 0)
final_audience_df = final_audience_df[final_audience_df["audience"] > 10000]
final_audience_df.fillna("0")
final_audience_df["audience"] = final_audience_df["audience"]

def make_number(x):
    try:
        return np.int(x.replace(",", ""))
    except:
        try:
            return np.int(x)
        except:
            return np.int(0)

for i in range(9):
    final_audience_df.ix[:, i] = final_audience_df.ix[:, i].apply(make_number)

def make_float(x):
    try:
        return np.float(x.replace("%", ""))
    except:
        try:
            return np.float(x)
        except:
            return np.float(0)

for i in range(9, 30):
    final_audience_df.ix[:, i] = final_audience_df.ix[:, i].apply(make_float)

final_audience_df = final_audience_df[final_audience_df["d7_audience"]<1500000].reset_index(drop=True)
boxoffice_df = pd.read_csv("data/boxoffice.csv", index_col=0)
final_audience_df.to_csv("final_audience.csv",encoding="utf-8")
total_movie_df = final_audience_df.merge(boxoffice_df,how="inner",on="movieCd")
total_movie_df = total_movie_df[list(boxoffice_df.columns)+list(final_audience_df.columns[1:])]
total_movie_df.to_csv("total_movie.csv",encoding="utf-8")

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

naver_df = pd.read_csv("data/total_movie.csv", index_col=0)

naver_df = naver_df[naver_df["movieNm"].str.contains("감독")==False]
naver_df = naver_df[naver_df["movieNm"].str.contains("확장")==False]
naver_df = naver_df.reset_index(drop=True)

def get_star_score_url(naver_df):
    url_df = pd.DataFrame(columns=["movieCd", "movieNm_x", "movieNm_y", "url"])
    for n, i in enumerate(list(naver_df["movieNm"])):

        try:
            data = ("http://movie.naver.com/movie/search/result.nhn?query={movieNm}&section=all&ie=utf8").format(
                movieNm=i)
            r = requests.get(data)
            dom = BeautifulSoup(r.content, "html.parser")
            dom1 = dom.select("p.result_thumb")
            open_year = naver_df["prdtYear"].apply(lambda x: str(x)[0:4])
            if dom1[-1] == dom1[0]:
                for i, data in enumerate(dom1):
                    url = data.select_one("a")["href"]
                    movieNm = dom.select("dt")[i].text
                    url_df.loc[len(url_df)] = [
                        naver_df["movieCd"][n],
                        naver_df["movieNm"][n],
                        movieNm,
                        url, ]
            elif dom1[-1] != dom1[0]:
                num = 0
                for j in range(0, len(dom.select("dd.etc")), 2):
                    year = dom.select("dd.etc")[j].text[-4::]
                    if (year == open_year[n]) & (num == j):
                        num += 10
                        url = dom1[int(j / 2)].select_one("a")["href"]
                        movieNm = dom.select("dt")[int(j / 2)].text
                        url_df.loc[len(url_df)] = [
                            naver_df["movieCd"][n],
                            naver_df["movieNm"][n],
                            movieNm,
                            url, ]
                    num += 2
        except:
            print(i)
    return url_df

url_df = get_star_score_url(naver_df)

url_df.to_csv("url.csv",encoding="utf-8")
url_df = pd.read_csv("data/url.csv",index_col=0)
print(url_df.head())

url_df["code"] = url_df["url"].apply(lambda x:np.int(x.split("=")[1]))
def get_star_score(url_df):
    star_score_df = pd.DataFrame(columns=["movieCd","movieNm","star_score","star_user_count"])#,"scene_score","scene_count"])
    for n, i in enumerate(list(url_df["code"])):
        try:
            data = ("http://movie.naver.com/movie/bi/mi/point.nhn?code={code}").format(code=i)
            r = requests.get(data)
            dom = BeautifulSoup(r.content, "html.parser")
            dom1 = dom.select_one("#beforePointArea")
            dom2 = dom1.select("em")[2:]
            star_user_count = dom2[-1].text
            star_score = ("").join([i.text for i in dom2 [:-1]])
            star_score_df.loc[len(star_score_df)] = [
                url_df["movieCd"][n],
                url_df["movieNm_x"][n],
                star_score,
                star_user_count,
            ]
        except:
            print(i)
    return star_score_df

star_score_df = get_star_score(url_df)

star_score_df["star_score"] = star_score_df["star_score"].astype(float)
star_score_df["star_user_count"] = star_score_df["star_user_count"].apply(lambda x: np.int(x.replace(",","")))
star_score_df.to_csv("star_score.csv",encoding="utf-8")

import requests
import pandas as pd
import numpy as np
import os
import time
import datetime
import json
from bs4 import BeautifulSoup

data_set = pd.read_csv("data/data_set.csv",index_col=0)
print(data_set)

data_set["openDt"] = data_set["openDt"].apply(lambda x: str(x)[4:6])

def month_change(i):
    if i == "01":
        return 'jan'
    elif i == '02':
        return 'feb'
    elif i == '03':
        return 'mar'
    elif i == '04':
        return 'apr'
    elif i == '05':
        return 'may'
    elif i == '06':
        return 'jun'
    elif i == '07':
        return 'jul'
    elif i == '08':
        return 'aug'
    elif i == '09':
        return 'sep'
    elif i == '10':
        return 'oct'
    elif i == '11':
        return 'nov'
    else:
        return 'dec'
data_set["openDt"] = data_set["openDt"].apply(month_change)

def grade_change(i):
    if i == '연소자관람가' or i == "모든 관람객이 관람할 수 있는 등급" or i == '전체관람가':
        return 'G'
    elif i == '12세관람가' or i == '12세이상관람가' or i == '국민학생관람불가' or i == '연소자관람불가' or i == '중학생이상관람가' or i == 'nan':
        return 'PG_13'
    elif i == '15세관람가' or i == '15세이상관람가' or i == '15세 미만인 자는 관람할 수 없는 등급 ' or i == '고등학생이상관람가':
        return 'R'
    elif i == '18세관람가' or i == '청소년관람불가':
        return 'NC_17'
    else:
        return i
data_set["watchGradeNm"] = data_set["watchGradeNm"].apply(grade_change)

data_set["showTm"] = data_set["showTm"].astype(int)

def time_change(i):
    if i < 90:
        return 'under_90'
    elif i >= 90 and i < 120:
        return '90_120'
    elif i >=120 and i < 150:
        return '120_150'
    else:
        return '150_up'

data_set["showTm"] = data_set["showTm"].apply(time_change)

def nation_change(i):
    if i == '중국' or i == '홍콩' or i == '대만':
        return 'china'
    elif i == '아이슬란드' or i == '우크라이나' or i == '체코' or i == '노르웨이' or i == ' 오스트리아' or i == '덴마크' or i == '러시아' or i == '이탈리아' or i == '벨기에' or i == '네덜란드' or i == '스페인' or i == '핀란드' or i == '스웨덴' or i == '스위스' or i == '영국' or i == '프랑스' or i == '헝가리' or i == '독일' or i == '아일랜드':
        return 'europe'
    elif i == '태국' or i == '싱가포르' or i == '호주' or i == '캐나다' or i == '페루' or i == '멕시코' or i == '이스라엘' or i == '뉴질랜드' or i == '아르헨티나' or i == '이란' or i == '남아프리카공화국' or i == '인도네시아' or i == '인도':
        return 'other_nation'
    elif i == '한국':
        return 'korea'
    elif i == '미국':
        return 'america'
    elif i == '일본':
        return 'japan'
data_set["repNationNm"] = data_set["repNationNm"].apply(nation_change)

def genre_change(i):
    if i == '드라마' or i == '멜로/로맨스':
        return 'drama_romance'
    elif i == '전쟁' or i == '액션':
        return 'war_action'
    elif i == '공포(호러)' or i == '미스터리':
        return 'horror_mystery'
    elif i == '범죄' or i == '스릴러':
        return 'crime_thriller'
    elif i == 'SF' or i == '판타지' or i == '어드벤처':
        return 'SF_fantasy_adventure'
    elif i == '애니메이션' or i == '가족':
        return 'family_animation'
    elif i == '코미디':
        return 'comedy'
    elif i == '다큐멘터리':
        return 'documentary'
    elif i == '공연' or i == '뮤지컬':
        return 'performance_musical'
    elif i == '사극' or i == '서부극(웨스턴)':
        return 'historical'
    else:
        return i

data_set["repGenreNm"] = data_set["repGenreNm"].apply(genre_change)

def company_change(i):
    if i == '소니픽쳐스릴리징월트디즈니스튜디오스코리아(주)' or i == '월트디즈니' or i == '월트디즈니컴퍼니코리아(주)' or i == '월트디즈니코리아㈜':
        return 'walt_disney'
    elif i == '이십세기폭스코리아(주)':
        return 'twentieth_century_fox'
    elif i == '씨제이이앤엠(주)':
        return 'cjenm'
    elif i == '(주)쇼박스':
        return 'showbox'
    elif i == '워너 브러더스 픽쳐스' or i == '워너브러더스 코리아(주)':
        return 'warnerbros'
    elif i == '㈜메가박스' or i == '메가박스(주)플러스엠':
        return 'megabox'
    elif i == '유니버셜 픽쳐스' or i == '유니버설픽쳐스인터내셔널 코리아(유)':
        return 'universal'
    elif i == '(주)넥스트엔터테인먼트월드(NEW)':
        return 'next'
    elif i == '(주)와우픽쳐스':
        return 'wowpictures'
    elif i == '롯데쇼핑㈜롯데엔터테인먼트':
        return 'lotte'
    else:
        return 'other_company'
data_set["companyNm"] = data_set["companyNm"].apply(company_change)

cate_set = data_set[['movieCd', 'director', 'openDt', 'prdtYear', 'repNationNm', 'repGenreNm', 'showTm', 'watchGradeNm', 'actor_1', 'actor_2', 'actor_3', 'companyNm','audience']]

def score_change(i):
    if i < 1000000:
        return 1
    elif i < 2000000 and i >= 1000000:
        return 2
    elif i < 3000000 and i >= 2000000:
        return 3
    elif i < 4000000 and i >= 3000000:
        return 4
    elif i < 5000000 and i >= 4000000:
        return 5
    elif i < 6000000 and i >= 5000000:
        return 6
    elif i < 7000000 and i >= 6000000:
        return 7
    elif i < 8000000 and i >= 7000000:
        return 8
    elif i < 9000000 and i >= 8000000:
        return 9
    else:
        return 10

cate_set["audience"] = cate_set["audience"].apply(score_change)

def score_change(i):
    if i < 20000:
        return 1
    elif i < 40000 and i >= 20000:
        return 2
    elif i < 60000 and i >= 40000:
        return 3
    elif i < 100000 and i >= 60000:
        return 4
    elif i < 150000 and i >= 100000:
        return 5
    elif i < 250000 and i >= 150000:
        return 6
    elif i < 500000 and i >= 250000:
        return 7
    elif i < 1000000 and i >= 500000:
        return 8
    elif i < 2000000 and i >= 1000000:
        return 9
    else:
        return 10

cate_set["audience"] = cate_set["audience"].apply(score_change)

print(cate_set)

d1 = cate_set[["director","audience"]]
o1 = cate_set[["openDt","audience"]]
p1 = cate_set[["prdtYear","audience"]]
n1 = cate_set[["repNationNm","audience"]]
g1 = cate_set[["repGenreNm","audience"]]
s1 = cate_set[["showTm","audience"]]
w1 = cate_set[["watchGradeNm","audience"]]
c1 = cate_set[["companyNm","audience"]]

d2 = d1.groupby("director").agg({"audience": np.mean})
o2 = o1.groupby("openDt").agg({"audience": np.mean})
p2 = p1.groupby("prdtYear").agg({"audience": np.mean})
n2 = n1.groupby("repNationNm").agg({"audience": np.mean})
g2 = g1.groupby("repGenreNm").agg({"audience": np.mean})
s2 = s1.groupby("showTm").agg({"audience": np.mean})
w2 = w1.groupby("watchGradeNm").agg({"audience": np.mean})
c2 = c1.groupby("companyNm").agg({"audience": np.mean})

d2 = d2.reset_index()
o2 = o2.reset_index()
p2 = p2.reset_index()
n2 = n2.reset_index()
g2 = g2.reset_index()
s2 = s2.reset_index()
w2 = w2.reset_index()
c2 = c2.reset_index()

d3 = cate_set.merge(d2,how='inner',on='director')
o3 = cate_set.merge(o2,how='inner',on='openDt')
p3 = cate_set.merge(p2,how='inner',on='prdtYear')
n3 = cate_set.merge(n2,how='inner',on='repNationNm')
g3 = cate_set.merge(g2,how='inner',on='repGenreNm')
s3 = cate_set.merge(s2,how='inner',on='showTm')
w3 = cate_set.merge(w2,how='inner',on='watchGradeNm')
c3 = cate_set.merge(c2,how='inner',on='companyNm')

d3 = d3.rename(columns={"audience_y":"director_score"})
o3 = o3.rename(columns={"audience_y":"openDt_score"})
p3 = p3.rename(columns={"audience_y":"prdtYear_score"})
n3 = n3.rename(columns={"audience_y":"repNationNm_score"})
g3 = g3.rename(columns={"audience_y":"repGenreNm_score"})
s3 = s3.rename(columns={"audience_y":"showTm_score"})
w3 = w3.rename(columns={"audience_y":"watchGradeNm_score"})
c3 = c3.rename(columns={"audience_y":"companyNm_score"})

d3 = d3[["movieCd","director_score"]]
o3 = o3[["movieCd","openDt_score"]]
p3 = p3[["movieCd","prdtYear_score"]]
n3 = n3[["movieCd","repNationNm_score"]]
g3 = g3[["movieCd","repGenreNm_score"]]
s3 = s3[["movieCd","showTm_score"]]
w3 = w3[["movieCd","watchGradeNm_score"]]
c3 = c3[["movieCd","companyNm_score"]]

a11 = cate_set[["movieCd","actor_1","audience"]]
a12 = cate_set[["movieCd","actor_2","audience"]]
a13 = cate_set[["movieCd","actor_3","audience"]]

a21 = a11.groupby("actor_1").agg({"audience": np.mean})
a22 = a12.groupby("actor_2").agg({"audience": np.mean})
a23 = a13.groupby("actor_3").agg({"audience": np.mean})

a21 = a21.reset_index()
a22 = a22.reset_index()
a23 = a23.reset_index()

a31 = cate_set.merge(a21,how='outer',on='actor_1')
a32 = cate_set.merge(a22,how='outer',on='actor_2')
a33 = cate_set.merge(a23,how='outer',on='actor_3')

a31 = a31.rename(columns = {"actor_1":"actor"})
a32 = a32.rename(columns = {"actor_2":"actor"})
a33 = a33.rename(columns = {"actor_3":"actor"})

a41 = a31[["actor","audience_y"]]
a42 = a32[["actor","audience_y"]]
a43 = a33[["actor","audience_y"]]

actor = pd.concat([a41,a42,a43])
actor = actor.groupby("actor").agg({"audience_y":np.mean})
actor = actor.reset_index()

a31 = a31[["movieCd","audience_y"]]
a32 = a32[["movieCd","audience_y"]]
a33 = a33[["movieCd","audience_y"]]

actor = a31.merge(a32,how="outer",on="movieCd")
actor = actor.merge(a33,how="outer",on="movieCd")
actor["actor_score"] = actor["audience_y"]+ actor["audience_y_x"] + actor["audience_y_y"]
actor = actor[["movieCd","actor_score"]]
actor = actor.fillna(0)
value_data_set = data_set.drop(['director', 'openDt', 'prdtYear', 'repNationNm', 'repGenreNm', 'showTm', 'watchGradeNm', 'actor_1', 'actor_2', 'actor_3', 'companyNm','audience'],axis=1)

dd_set = value_data_set.merge(d3,how="outer",on="movieCd")
dd_set = dd_set.merge(o3,how="outer",on="movieCd")
dd_set = dd_set.merge(p3,how="outer",on="movieCd")
dd_set = dd_set.merge(n3,how="outer",on="movieCd")
dd_set = dd_set.merge(g3,how="outer",on="movieCd")
dd_set = dd_set.merge(s3,how="outer",on="movieCd")
dd_set = dd_set.merge(w3,how="outer",on="movieCd")
dd_set = dd_set.merge(c3,how="outer",on="movieCd")
dd_set = dd_set.merge(actor,how="outer",on="movieCd")

dd_set["director_score"] = round(dd_set["director_score"],1)
dd_set["openDt_score"] = round(dd_set["openDt_score"],1)
dd_set["prdtYear_score"] = round(dd_set["prdtYear_score"],1)
dd_set["repNationNm_score"] = round(dd_set["repNationNm_score"],1)
dd_set["repGenreNm_score"] = round(dd_set["repGenreNm_score"],1)
dd_set["showTm_score"] = round(dd_set["showTm_score"],1)
dd_set["watchGradeNm_score"] = round(dd_set["watchGradeNm_score"],1)
dd_set["companyNm_score"] = round(dd_set["companyNm_score"],1)
dd_set["actor_score"] = round(dd_set["actor_score"],1)

dd_set = dd_set.drop(["movieCd","movieNm"],axis=1)
dd_set = dd_set.fillna(0)
print(dd_set)

dd_set.to_csv("value_data.csv",encoding="utf-8")

import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

data = pd.read_csv("data/data.csv", index_col=0)
dd_set = pd.read_csv("data/value_data.csv", index_col=0)
modeling_data = pd.concat([dd_set,data["audience"]],axis=1)

X = modeling_data.ix[:,:-1]
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)

dfX0 = pd.DataFrame(X_scaled, columns=X.columns)
dfX = sm.add_constant(dfX0)
dfy = pd.DataFrame(modeling_data.ix[:,-1], columns=["audience"])
d_df = pd.concat([dfX, dfy], axis=1)
print(d_df.head())

d_df = d_df[["const","star_score","preview_audience","d1_audience","d3_audience","d4_audience","d5_audience","d6_audience","d1_screen","d4_screen","d7_screen","d2_seat","d7_seat","director_score","actor_score","audience"]]

from sklearn.model_selection import train_test_split
X = d_df.ix[:,:-1]
y = d_df.ix[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = sm.OLS(d_df.ix[:,-1],d_df.ix[:,1:-1])
result = model.fit()
print(result.summary())

model = sm.OLS(y_train,X_train)
result = model.fit()
print(result.summary())

remove_column_list = ["d2_screen","d3_screen","d5_show","d7_show", "showTm_score","d3_seat","openDt_score"]
d_df = d_df.drop(remove_column_list, axis=1)

model = sm.OLS(d_df.ix[:, -1], d_df.ix[:, 1:-1])
result = model.fit()
print(result.summary())

remove_column_list = ["repGenreNm_score"]
d_df = d_df.drop(remove_column_list, axis=1)

model = sm.OLS(d_df.ix[:, -1], d_df.ix[:, 1:-1])
result = model.fit()
print(result.summary())

movie_columns = list(d_df.columns)[:-1]

formula_str = "audience ~ " + " + ".join(movie_columns)
model = sm.OLS.from_formula(formula_str, data=d_df)
result = model.fit()
table_anova = sm.stats.anova_lm(result)
table_anova

remove_column_list = ["d7_audience","d5_screen","d3_show","d4_show","d7_show","d2_seat","d3_seat","d4_seat","d5_seat","openDt_score","prdtYear_score","showTm_score","repGenreNm_score","watchGradeNm_score","repNationNm_score","companyNm_score"]
d_df = d_df.drop(remove_column_list, axis=1)

model = sm.OLS(d_df.ix[:, -1], d_df.ix[:, 1:-1])
result = model.fit()
print(result.summary())

remove_column_list = ['d7_audience']
d_df = d_df.drop(remove_column_list, axis=1)

model = sm.OLS(d_df.ix[:, -1], d_df.ix[:, 1:-1])
result = model.fit()
print(result.summary())

model = LinearRegression()
model.fit(d_df.ix[:, :-1], d_df.ix[:, -1])

def view_result(number):
    a = d_df[d_df["audience"].index==number]
    print(model.predict(a.ix[:,:-1]))
    print(d_df[d_df["audience"].index==number]["audience"])

    view_result(1635)

#end_time = time.time() - start_time
#print('연산시간:', end_time)
#연산시간 : 30분 18초